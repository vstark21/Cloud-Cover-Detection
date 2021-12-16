import torch
import torch.nn as nn

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