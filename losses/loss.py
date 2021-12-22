from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.jaccard_loss import *
from losses.dice_loss import *
from losses.filtered_jaccard_loss import *

class CloudLoss(nn.Module):
    def __init__(
        self, 
        bce_lw=1.0, 
        dice_lw=1.0, 
        jacc_lw=1.0,
        fjacc_lw=1.0,
    ):
        super().__init__()
        s = (bce_lw + dice_lw + jacc_lw + fjacc_lw)
        self.bce_lw = bce_lw / s
        self.dice_lw = dice_lw / s
        self.jacc_lw = jacc_lw / s
        self.fjacc_lw = fjacc_lw / s

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.jacc_loss = JaccardLoss()
        self.filtered_jacc_loss = FilteredJaccardLoss()

    def __call__(self, preds, targets):
        jacc = self.jacc_loss(preds, targets)
        bce = self.bce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        fjacc = self.filtered_jacc_loss(preds, targets)
        loss = self.bce_lw * bce + \
                self.dice_lw * dice + \
                self.jacc_lw * jacc + \
                self.fjacc_lw * fjacc

        return loss, bce, dice
