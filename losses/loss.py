import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.jaccard_loss import *
from losses.dice_loss import *
from losses.filtered_jaccard_loss import *

class CloudLoss(nn.Module):
    def __init__(
        self, 
        loss_cfg: dict
    ):
        super().__init__()
        self.out_weights = loss_cfg.pop('out_weights')
        loss_weights = loss_cfg.pop('loss_weights')
        s = sum(loss_weights.values())
        losses = dict()
        for name, type in loss_cfg['losses']:
            losses[name] = globals()[type]()
            loss_weights[name] /= s
        self.losses = losses
        self.loss_weights = loss_weights

    def __call__(self, preds_dict, targets):
        loss_dict = dict()
        for name, loss in self.losses.items():
            loss_dict[name] = 0
            for key, preds in preds_dict.items():
                loss_dict[name] += (
                    self.out_weights[key] * loss(preds, targets)
                )
        for name, loss in loss_dict.items():
            loss_dict['loss'] += (
                self.loss_weights[name] * loss
            )
            
        return loss_dict
