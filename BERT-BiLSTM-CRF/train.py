import os
import torch
import config
from model import BertNER
import logging
from tqdm import tqdm
import torch.nn as nn
from transformers import BertTokenizer
import torch
from metrics import f1_score,bad_case

def evaluate(dev_loader,model,mode='dev'):
    model.eval()
    if mode=='test':
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, skip_special_tokens=True)
    id2label=config.id2label
    true_tags=[]
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx,batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks=batch_data.gt(0)
            label_masks=batch_tags.gt(-1)
            loss,batch_output = model([batch_data, batch_token_starts], attention_mask=batch_masks, labels=batch_tags)
            dev_losses+=loss.item()
            batch_output=model.crf.decode(batch_output,mask=label_masks)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
        assert len(pred_tags)==len(true_tags)
        if mode=='test':
            assert len(sent_data) == len(true_tags)
            # logging loss, f1 and report
        metrics = {}
        if mode == 'dev':
            f1 = f1_score(true_tags, pred_tags, mode)
            metrics['f1'] = f1
        else:
            bad_case(true_tags, pred_tags, sent_data)
            f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
            metrics['f1_labels'] = f1_labels
            metrics['f1'] = f1
        metrics['loss'] = float(dev_losses) / len(dev_loader)
        return metrics



def train(train_loader,dev_loader,model,optimizer,scheduler,model_dir):
    if os.path.exists(config.model_dir) and config.load_before:
        state = torch.load(config.model_dir)
        model = BertNER(config)
        model.to(config.device)
        model.load_state_dict(state['model_state'])
        print('train阶段加载模型完成')
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(1,config.epoch_num+1):
        model.train()
        train_losses=0
        for idx,batch_samples in enumerate(tqdm(train_loader)):
            batch_data,batch_token_starts,batch_labels=batch_samples
            batch_masks=batch_data.gt(0)
            loss=model([batch_data,batch_token_starts],attention_mask=batch_masks, labels=batch_labels)[0]
            train_losses += loss.item()
            # clear previous gradients, compute gradients of all variables wrt loss
            model.zero_grad()
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            # performs updates using calculated gradients
            optimizer.step()
            scheduler.step()
        train_loss=float(train_losses)/len(train_loader)
        logging.info('Epoch: {},train loss: {}'.format(epoch,train_loss))
        val_metrics = evaluate(dev_loader, model, mode='dev')
        val_f1 = val_metrics['f1']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))
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










