import torch
import torch.nn as nn

class BceLoss(nn.Module):
    def __init__(
        self,
        from_logits=True
    ):
        super().__init__()
        self.loss = None
        if from_logits:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.BCELoss()
        
    
    def __call__(self, preds, targets):
        pflat = preds.view(-1)
        tflat = targets.view(-1)
        loss = self.loss(pflat, tflat)
        return loss
