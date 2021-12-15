import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class JaccardLoss(nn.Module):
    def __init__(
        self,
        smooth=0,
        from_logits=True,
        eps=1e-7
    ):
        super().__init__()
        self.eps = eps
        self.smooth = smooth
        self.from_logits = from_logits
    
    def __call__(self, preds, targets):
        if self.from_logits:
            preds = torch.sigmoid(preds)
            
        preds = torch.round(preds)
        intersection = torch.logical_and(preds, targets).sum() + self.smooth
        union = torch.logical_or(preds, targets).sum() + self.eps + self.smooth

        return 1 - (intersection / union)

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
        self.jacc_loss = JaccardLoss(smooth=1)

    def __call__(self, preds, targets, smooth=1):
        bce = self.bce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        jacc = self.jacc_loss(preds, targets)
        loss = self.bce_lw * bce + \
                self.dice_lw * dice + \
                self.jacc_lw * jacc

        return loss, bce, dice
