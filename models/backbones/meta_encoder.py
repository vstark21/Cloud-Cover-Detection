import torch
import torch.nn as nn
from layers import UpBlock

class MetaEncoder(nn.Module):
    def __init__(self, input_size):
        super(MetaEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True),
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.layer3 = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), 1, 32, 32)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
