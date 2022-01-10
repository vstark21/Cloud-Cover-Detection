import torch
import torch.nn as nn
from build_utils import build_backbone, build_head
from layers import UpBlock

class SeMask(nn.Module):
    def __init__(self, cfg):
        super(SeMask, self).__init__()

        self.backbone = build_backbone(cfg['backbone'])
        self.decode_head = build_head(cfg['decode_head'])

        # self.decode_up = nn.Sequential(
        #     UpBlock(512, 128, (256, 256)),
        #     UpBlock(128, 32, (512, 512))
        # )
        # self.decode_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)

        x_1 = self.decode_head(x)
        # x_1 = self.decode_up(x_1)
        # out_1 = self.decode_conv(x_1)
        print(x.shape, x_1.shape)
        return dict(out=x_1)
