import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.jaccard_loss import *
from losses.dice_loss import *
from losses.filtered_jaccard_loss import *

class CloudLoss(nn.Module):
    def __init__(
        self, 
        out_weights: dict,
        loss_weights: list
    ):
        super().__init__()
        s = sum(loss_weights)
        self.bce_lw = loss_weights[0] / s
        self.dice_lw = loss_weights[1] / s
        self.jacc_lw = loss_weights[2] / s
        self.fjacc_lw = loss_weights[3] / s
        self.out_weights = out_weights

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.jacc_loss = JaccardLoss()
        self.filtered_jacc_loss = FilteredJaccardLoss()

    def __call__(self, preds_dict, targets):
        loss, bce, dice = 0, 0, 0

        for key, preds in preds_dict.items():
            cur_jacc = self.jacc_loss(preds, targets)
            cur_bce = self.bce_loss(preds, targets)
            cur_dice = self.dice_loss(preds, targets)
            cur_fjacc = self.filtered_jacc_loss(preds, targets)
            cur_loss = self.bce_lw * cur_bce + \
                    self.dice_lw * cur_dice + \
                    self.jacc_lw * cur_jacc + \
                    self.fjacc_lw * cur_fjacc
            
            loss += (self.out_weights[key] * cur_loss)
            bce += (self.out_weights[key] * cur_bce)
            dice += (self.out_weights[key] * cur_dice)

        return loss, bce, dice
