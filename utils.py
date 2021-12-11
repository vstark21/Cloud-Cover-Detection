import os
import torch
import random
import numpy as np
from loguru import logger


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
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
    

def save_model_weights(model, filename, verbose=1, folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        logger.info(f"Saving weights to {os.path.join(folder, filename)}")
    torch.save(model.state_dict(), os.path.join(folder, filename))


def load_model_weights(model, filename, verbose=1, folder=""):
    """
    Loads the weights of a PyTorch model.

    Args:
        model (torch model): Model to load the weights.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        folder (str, optional): Folder to load from. Defaults to "".
    Returns:
        model (torch model): model with weights loaded
    """
    if verbose:
        logger.info(f"Loading weights from {os.path.join(folder, filename)}")
    if not os.path.exists(os.path.join(folder, filename)):
        raise ValueError(f"{filename} doesn't exist at {os.path.join(folder, filename)}")
    model.load_state_dict(torch.load(os.path.join(folder, filename)))
    return model


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

def jaccard_score(pred, true):
    pred = torch.sigmoid(pred)
    pred = torch.round(pred)
    
    intersection = torch.logical_and(pred, true).sum()
    union = torch.logical_or(pred, true).sum()

    return intersection / union