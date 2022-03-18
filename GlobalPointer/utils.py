import logging
import numpy as np
from sklearn.model_selection import train_test_split
import config

def set_logger(log_path):
    logger = logging.getLogger()#用logging.getLogger(name)方法进行初始化
    logger.setLevel(logging.INFO)#设置级别

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)#地址
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def val_split(dataset_dir):
    """split one dev set without k-fold"""
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"][:50]
    labels = data["labels"][:50]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=config.val_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev