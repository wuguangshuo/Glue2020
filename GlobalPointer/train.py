import os
import config
from model import GlobalPointer
import logging
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

log_dir='./runs/'
writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))


def evaluate(val_loader,model):
    val_losses, val_presion, val_f1 ,val_recall= 0, 0, 0,0
    with torch.no_grad():
        for idx,batch_samples in enumerate(val_loader):
            batch_data, batch_attention_mask, batch_token_type, batch_labels=batch_samples
            logits=model(batch_data,attention_mask=batch_attention_mask,token_type_ids=batch_token_type ,labels=batch_labels)
            loss = loss_fun(logits, batch_labels)
            f1, precision, recall = get_evaluate_fpr(logits, batch_labels)
            val_losses+=loss.item()
            val_f1+=f1
            val_presion+=precision
            val_recall+=recall
    val_losses=val_losses/len(val_loader)
    val_f1=val_f1/len(val_loader)
    val_presion=val_presion/len(val_loader)
    val_recall=val_recall/len(val_loader)

    return val_losses,val_f1,val_presion,val_recall



def train(train_loader,val_loader,model,optimizer,scheduler,model_dir):
    if os.path.exists(config.model_dir) and config.load_before:
        state = torch.load(config.model_dir)
        model = GlobalPointer(config)
        model.to(config.device)
        model.load_state_dict(state['model_state'])
        print('train阶段加载模型完成')
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(1,config.epoch_num+1):
        model.train()
        train_losses,train_presion,train_f1=0,0,0
        for idx,batch_samples in enumerate(tqdm(train_loader)):
            batch_data, batch_attention_mask, batch_token_type, batch_labels=batch_samples
            logits=model(batch_data,attention_mask=batch_attention_mask,token_type_ids=batch_token_type ,labels=batch_labels)
            loss = loss_fun(logits, batch_labels)
            presion = get_sample_precision(logits, batch_labels)
            f1 = get_sample_f1(logits, batch_labels)
            train_losses += loss.item()
            train_presion+=presion.item()
            train_f1+=f1.item()
            # clear previous gradients, compute gradients of all variables wrt loss
            model.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            # performs updates using calculated gradients
            optimizer.step()
            scheduler.step()
        train_loss=float(train_losses)/len(train_loader)
        train_presion = float(train_presion) / len(train_loader)
        train_f1 = float(train_f1) / len(train_loader)
        logging.info("Epoch: {},train loss: {}, train f1: {},train presion: {}".format(epoch, train_loss,train_f1,train_presion))

        val_loss,val_f1,val_presion,val_recall= evaluate(val_loader, model, mode='val')

        logging.info("Epoch: {},vaild loss: {}, vaild f1: {},vaild presion: {},vaild recall: {}".format(epoch, val_loss, val_f1,
                                                                                       val_presion,val_recall))

        writer.add_scalar('Training/training loss',train_loss ,epoch)
        writer.add_scalar('Training/training f1', train_f1, epoch)
        writer.add_scalar('Validation/loss', val_loss, epoch)
        writer.add_scalar('Validation/acc', val_f1, epoch)

        improve_f1 = val_f1 - best_val_f1
        state = {}
        if improve_f1 >1e-5:
            best_val_f1 = val_f1
            state['model_state'] = model.state_dict()
            torch.save(state,config.model_dir)
            logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
    logging.info("Training Finished!")

def get_evaluate_fpr(y_pred, y_true):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    pred = []
    true = []
    for b, l, start, end in zip(*np.where(y_pred > 0)):
        pred.append((b, l, start, end))
    for b, l, start, end in zip(*np.where(y_true > 0)):
        true.append((b, l, start, end))

    R = set(pred)
    T = set(true)
    X = len(R & T)
    Y = len(R)
    Z = len(T)
    f1, precision, recall = 2 * X / (Y + Z + 1e-12), X / (Y + 1e-12), X / (Z + 1e-12)
    # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size*ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size*ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss

def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = ( y_pred - (1 - y_true) * 1e12 )  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()

def get_sample_f1(y_pred, y_true):
    y_pred = torch.gt(y_pred, 0).float()
    return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

def get_sample_precision(y_pred, y_true):
    y_pred = torch.gt(y_pred, 0).float()
    return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)






