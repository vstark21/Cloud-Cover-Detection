import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.jaccard_loss import *
from losses.dice_loss import *
from losses.filtered_jaccard_loss import *
from losses.bce_loss import *

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
        for name, type in loss_cfg['losses'].items():
            losses[name] = globals()[type]()
            loss_weights[name] /= s
        self.losses = losses
        self.loss_weights = loss_weights

    def __call__(
        self, 
        preds_dict: dict, 
        targets: torch.Tensor
    ):
        loss_dict = dict()
        for name, loss in self.losses.items():
            loss_dict[name] = 0
            for key, preds in preds_dict.items():
                loss_dict[name] += (
                    self.out_weights[key] * loss(preds, targets)
                )
        loss_dict['loss'] = 0
        for name, weight in self.loss_weights.items():
            loss_dict['loss'] += (
                weight * loss_dict[name]
            )

        return loss_dict
