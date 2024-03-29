import torch
import torch.nn as nn
from build_utils import build_backbone, build_head
from layers import UpBlock

class SeMask(nn.Module):
    def __init__(self, cfg):
        super(SeMask, self).__init__()

        self.backbone = build_backbone(cfg['backbone'])
        self.decode_head = build_head(cfg['decode_head'])

        self.decode_up = nn.Sequential(
            UpBlock(512, 128, (256, 256)),
            UpBlock(128, 32, (512, 512)),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.auxiliary_up = nn.Sequential(
            UpBlock(512, 128, (256, 256)),
            UpBlock(128, 32, (512, 512)),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)

        x_1, x_2 = self.decode_head(x)
        out_1 = self.decode_up(x_1)

        if not self.training:
            return dict(out=out_1)

        out_2 = self.auxiliary_up(x_2)
        return dict(out=out_1, aux_out=out_2)
