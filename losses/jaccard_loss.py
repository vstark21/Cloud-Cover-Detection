import torch
import torch.nn as nn

class JaccardLoss(nn.Module):
    def __init__(
        self,
        from_logits=True,
        eps=1e-7
    ):
        super().__init__()
        self.eps = eps
        self.from_logits = from_logits
    
    def __call__(self, preds, targets):
        if self.from_logits:
            preds = torch.sigmoid(preds)
        preds = torch.round(preds)
        intersection = torch.logical_and(preds, targets).sum()
        union = torch.logical_or(preds, targets).sum() + self.eps

        return 1 - (intersection / union)