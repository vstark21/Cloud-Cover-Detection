import os
import torch
import torch.nn as nn
import random
import numpy as np
from loguru import logger

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def seed_everything(
    seed: None or int = 42,
):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (None or int): Number of the seed.
    """
    if not seed:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def count_parameters(model, all=False):
    """
    Counts the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def save_checkpoint(
    filename,
    **kwargs,
):
    logger.info(f"Saving state to {filename}")
    checkpoint = {}
    for k, v in kwargs.items():
        checkpoint[k] = v
    torch.save(checkpoint, filename)

def load_checkpoint(
    filename,
):
    logger.info(f"Loading weights from {filename}")
    if not os.path.exists(filename):
        raise ValueError(f"{filename} doesn't exist at {filename}")
    checkpoint = torch.load(filename)
    return checkpoint


def format_time(seconds):
    """
    Formates time in human readable form

    Args:
        seconds: seconds passed in a process
    Return:
        formatted string in form of MM:SS or HH:MM:SS
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    result = ''
    _h = ('0' + str(h)) if h < 10 else str(h)
    result += (_h + ' hr ') if h > 0 else ''
    _m = ('0' + str(m)) if m < 10 else str(m)
    result += (_m + ' min ') if m > 0 else ''
    _s = ('0' + str(s)) if s < 10 else str(s)
    result += (_s + ' sec')
    return result

@torch.no_grad()
def jaccard_score(pred, true, eps=1e-7, from_logits=True):
    if from_logits:
        pred = torch.sigmoid(pred)
    pred = torch.round(pred)
    intersection = torch.logical_and(pred, true).sum()
    union = torch.logical_or(pred, true).sum() + eps

    return intersection / union
