import torch
import torch.nn as nn

class DiceLoss(nn.Module):
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
        pflat = preds.view(-1)
        tflat = targets.view(-1)
        intersection = (pflat * tflat).sum()
        score = (2. * intersection + self.smooth) / (pflat.sum() + tflat.sum() + self.smooth + self.eps)
        return 1 - score