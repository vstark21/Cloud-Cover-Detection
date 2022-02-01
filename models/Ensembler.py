import torch
import torch.nn as nn

class Nugget(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_act=True):
        super(Nugget, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=kernel_size)
        self.act = None
        if use_act:
            self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv2d(x)
        if self.act:
            x = self.act(x)
        return x

class Ensembler(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1
    ):
        super(Ensembler, self).__init__()
        self.nuggets = nn.Sequential(
            Nugget(in_channels, 16, 1),
            Nugget(16, in_channels, 1, use_act=False),
        )
        self.act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        identity = x
        x = self.nuggets(x)
        x += identity
        x = self.act(x)
        out = self.out_conv(x)

        return dict(out=out)
