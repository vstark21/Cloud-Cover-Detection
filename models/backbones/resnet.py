import torch
import torch.nn as nn

BACKBONE_CHANNELS = {
    'resnet18': (512, 256, 128, 64, 64),
    'resnet34': (512, 256, 128, 64, 64),
    'resnet50': (2048, 1024, 512, 256, 64),
    'resnet101': (2048, 1024, 512, 256, 64),
    'resnet152': (2048, 1024, 512, 256, 64),
    'resnext50_32x4d': (2048, 1024, 512, 256, 64),
    'resnext101_32x8d': (2048, 1024, 512, 256, 64),
    'resnext101_32x16d': (2048, 1024, 512, 256, 64),
    'resnext101_32x32d': (2048, 1024, 512, 256, 64),
    'resnext101_32x48d': (2048, 1024, 512, 256, 64),
}

class ResNet(nn.Module):
    def __init__(
        self,
        variant: str,
        n_channels: int,
    ):
        super(ResNet, self).__init__()
        self.backbone = torch.hub.load("pytorch/vision", variant, pretrained=False)
        self.out_channels = BACKBONE_CHANNELS[variant]
        inplanes = self.out_channels[-1]
        self.backbone.conv1 = nn.Conv2d(
            n_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        del self.backbone.fc

    def forward(self, x):
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)

        x1 = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x1)

        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        return (x4, x3, x2, x1, x0)
