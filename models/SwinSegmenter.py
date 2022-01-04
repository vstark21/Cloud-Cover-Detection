import torch
import torch.nn as nn
from backbones import SwinTransformer
from heads import UPerHead, FCNHead
from layers import UpBlock

class SwinSegmenter(nn.Module):
    def __init__(self, cfg):
        super(SwinSegmenter, self).__init__()

        self.backbone = SwinTransformer(**cfg['backbone'])
        self.decode_head = UPerHead(**cfg['decode_head'])
        self.auxiliary_head = FCNHead(**cfg['auxiliary_head'])

        self.decode_up = nn.Sequential(
            UpBlock(512, 128, (256, 256)),
            UpBlock(128, 32, (512, 512))
        )
        self.auxiliary_up = nn.Sequential(
            UpBlock(512, 256, (128, 128)),
            UpBlock(256, 128, (256, 256)),
            UpBlock(128, 32, (512, 512))
        )
        self.decode_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.auxiliary_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)

        x_1 = self.decode_head(x)
        x_1 = self.decode_up(x_1)
        out_1 = self.decode_conv(x_1)

        x_2 = self.auxiliary_head(x)
        x_2 = self.auxiliary_up(x_2)
        out_2 = self.auxiliary_conv(x_2)

        return dict(out=out_1, aux_out=out_2)
