import torch
import torch.nn as nn
import torch.nn.functional as F

class CloudLoss(nn.Module):
    def __init__(self, l1=1.0, l2=1.0):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, preds, targets, smooth=1):
        inputs = torch.sigmoid(preds)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        preds = preds.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='mean')
        
        loss = self.l1 * bce + self.l2 * dice_loss
        
        return loss