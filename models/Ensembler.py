import torch
import torch.nn as nn

class Nugget(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, padding=0):
        super(Nugget, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                padding=padding)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv2d(x)
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
            Nugget(3, in_channels, 16, padding=1),
            Nugget(1, 16, in_channels),
        )
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.nuggets(x)
        out = self.out_conv(x)

        return dict(out=out)
