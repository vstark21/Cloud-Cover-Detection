import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class CloudLoss(nn.Module):
    def __init__(
        self, 
        bce_lw=1.0, 
        dice_lw=1.0, 
        jacc_lw=1.0
    ):
        super().__init__()
        self.bce_lw = bce_lw / (bce_lw + dice_lw + jacc_lw)
        self.dice_lw = dice_lw / (bce_lw + dice_lw + jacc_lw)
        self.jacc_lw = jacc_lw / (bce_lw + dice_lw + jacc_lw)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary', smooth=1)
        self.jacc_loss = smp.losses.JaccardLoss(mode='binary', smooth=1)

    def __call__(self, preds, targets, smooth=1):
        bce = 0
        dice = self.dice_loss(preds, targets)
        jacc = self.jacc_loss(preds, targets)
        loss = self.bce_lw * bce + \
                self.dice_lw * dice + \
                self.jacc_lw * jacc

        return loss, bce, dice
