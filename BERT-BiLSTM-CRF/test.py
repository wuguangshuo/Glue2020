import numpy as np
import torch
from model import BertNER
import config
from data_utils import NERDataset
import logging
from torch.utils.data import DataLoader
from train import evaluate
def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    test_dataset = NERDataset(word_test, label_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        state = torch.load(config.model_dir)
        model = BertNER(config)
        model.to(config.device)
        model.load_state_dict(state['model_state'])
        print('加载模型完成')
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        print('加载模型失败')
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))
        print("f1 score of {}: {}".format(label, val_f1_labels[label]))
