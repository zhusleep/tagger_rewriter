import torch

import random
import os
import numpy as np
import logging


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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
