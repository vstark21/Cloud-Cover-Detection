import torch
import torch.nn as nn

class Nugget(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Nugget, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=kernel_size)
        self.act = nn.SiLU()   

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class Ensembler(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1
    ):
        super(Ensembler, self).__init__()
        # self.nuggets = nn.Sequential(
        #     Nugget(in_channels, 16, 1),
        #     Nugget(16, in_channels, 1),
        # )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        # x = self.nuggets(x)
        x = self.conv1(x)

        return dict(out=x)
