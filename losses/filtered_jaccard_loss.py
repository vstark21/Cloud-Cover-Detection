import torch
import torch.nn as nn
from losses.jaccard_loss import JaccardLoss

def low_and_high_pass(
    y: torch.Tensor,
    cutoff: float,
    steepness: float,
):
    S = y.sum(dim=1)

    lp = 1 / (1 + torch.exp(steepness * (S - cutoff)))
    hp = 1 / (1 + torch.exp(steepness * (-S + cutoff)))
    return lp, hp


# Taken from https://arxiv.org/pdf/2001.08768.pdf
class FilteredJaccardLoss(nn.Module):
    def __init__(
        self,
        from_logits=True,
        eps=1e-7,
        m=1000,
        pc=0.5,
        kg=1,
        kj=1,
        mode='inv', # 'inv' or 'bce'
    ):
        super().__init__()
        self.eps = eps
        self.from_logits = from_logits
        self.m = m
        self.pc = pc
        self.kg = kg
        self.kj = kj
        self.mode = mode

        self.jl = JaccardLoss(from_logits=False)
    
    def __call__(self, y, t):
        if self.from_logits:
            y = torch.sigmoid(y)
        batch_size = y.shape[0]
        y = y.view(batch_size, -1)
        t = t.view(batch_size, -1)

        lp, hp = low_and_high_pass(y, self.pc, self.m)
        jl = self.jl(y, t)
        gl = 0
        if self.mode == 'inv':
            gl = self.jl((1 - y), (1 - t))
        # TODO: implement the other mode (bce)
        loss = (
            self.kg * gl * lp + self.kj * jl * hp
        )
        return loss
        
        