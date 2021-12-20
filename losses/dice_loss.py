import torch
import torch.nn as nn

class DiceLoss(nn.Module):
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
        pflat = preds.view(-1)
        tflat = targets.view(-1)
        intersection = (pflat * tflat).sum()
        score = (2. * intersection) / (pflat.sum() + tflat.sum() + self.eps)
        return 1 - score