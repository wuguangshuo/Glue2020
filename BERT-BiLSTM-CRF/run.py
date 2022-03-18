import utils
import logging
import config
from torch.utils.data import DataLoader
from data_utils import Processor,NERDataset
from model import BertNER
from transformers.optimization import AdamW,get_cosine_schedule_with_warmup
from train import train
from test import test
if __name__=='__main__':
    utils.set_logger(config.log_dir)
    logging.info('device: {}'.format(config.device))

    #处理json数据，建立词表
    processor = Processor(config)
    processor.data_process()

    logging.info('----------process done!-----------')
    #将训练集划分训练集和验证集
    word_train, word_dev, label_train, label_dev = utils.dev_split(config.train_dir)
    train_dataset=NERDataset(word_train,label_train,config)
    dev_dataset = NERDataset(word_dev, label_dev, config)
    logging.info("--------Dataset Build!--------")
    train_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)

    model=BertNER(config)
    model.to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=config.lr)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)
    test()

