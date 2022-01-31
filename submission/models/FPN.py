import torch
import torch.nn as nn
from build_utils import build_backbone, build_head

class FPN(nn.Module):
    def __init__(
        self,
        cfg: dict
    ):
        super().__init__()
        
        self.backbone = build_backbone(cfg['backbone'])
        self.decode_head = build_head(cfg['head'], encoder_channels=self.backbone.out_channels)

    def forward(self, x):
        encodes = self.backbone(x)
        out = self.decode_head(x, encodes)

        return dict(out=out)
