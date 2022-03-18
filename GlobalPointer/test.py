import numpy as np
import torch
from model import GlobalPointer
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
        model = GlobalPointer(config)
        model.to(config.device)
        model.load_state_dict(state['model_state'])
        print('加载模型完成')
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        print('加载模型失败')
        return
    test_loss, test_f1, test_presion, test_recall = evaluate(test_loader, model)
    print("test loss: {}, test f1: {},test presion: {},test recall: {}".format(test_loss, test_f1,test_presion,test_recall))
if __name__=='__main__':
    test()