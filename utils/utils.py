import errno
import json
import logging
import os
import os.path as osp
import random
import shutil
import sys
from config import CONFIG
import numpy as np
import torch

from copy import deepcopy
from typing import Any

from torch import nn


def build_model_name():
    model_name = f'{CONFIG.DATA.DATASET}_{CONFIG.TRAIN.MAX_EPOCH}_{CONFIG.TRAIN.OPTIMIZER.LR}_{CONFIG.DATA.TRAIN_BATCH}_{CONFIG.MODEL.NAME}_{CONFIG.TRAIN.TRAIN_MODE}'
    model_name += '.pth'
    return model_name

def is_list_or_tuple(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def clones(module: Any, N: int) -> nn.ModuleList:
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

'''
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
'''


def get_logger(fpath, name=''):
    # Creat logger
    logger = logging.getLogger(name)
    level = logging.INFO 
    logger.setLevel(level=level)

    # Output to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=level) 
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Output to file
    if fpath is not None:
        mkdir_if_missing(os.path.dirname(fpath))
    file_handler = logging.FileHandler(fpath, mode='w')
    file_handler.setLevel(level=level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    return logger