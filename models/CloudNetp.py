import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Conv, ResBlock
from build_utils import build_backbone

class ContractionBlock(nn.Module):
    def __init__(self, ni, residual=False):
        super(ContractionBlock, self).__init__()

        if residual:
            self.conv1 = ResBlock(ni, ni, kernel_size=3, stride=1, padding=1)
            self.conv2 = ResBlock(ni, 2 * ni, kernel_size=1, stride=1, padding=0)
            self.conv3 = ResBlock(2 * ni, 2 * ni, kernel_size=3, stride=1, padding=1)

            self.conv4 = ResBlock(ni, ni, kernel_size=1, stride=1, padding=0)

        else:
            self.conv1 = Conv(ni, ni, kernel_size=3, stride=1, padding=1)
            self.conv2 = Conv(ni, 2 * ni, kernel_size=1, stride=1, padding=0)
            self.conv3 = Conv(2 * ni, 2 * ni, kernel_size=3, stride=1, padding=1)

            self.conv4 = Conv(ni, ni, kernel_size=1, stride=1, padding=0)

        self.l5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)

        x2 = self.conv4(x)
        x2 = torch.cat([x2, x], dim=1)
        
        x = x1 + x2
        x = self.l5(x)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, ni, n, residual=False):
        super(FeedForwardBlock, self).__init__()

        self.poolings = []
        for i in range(n):
            s = 2 ** (n - i)
            layer = nn.MaxPool2d(kernel_size=3, stride=s, padding=1)

            self.poolings.append(layer)
        
        if residual:
            self.conv = ResBlock(ni, ni, kernel_size=3, stride=1, padding=1)
        else:
            self.conv = Conv(ni, ni, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        n = len(self.poolings)
        out = x[n]
        for i in reversed(range(n)):
            s = n - i
            concat_list = []
            for _ in range(2 ** s):
                concat_list.append(x[i])
            xi = torch.cat(concat_list, dim=1)
            xi = self.poolings[i](xi)
            out = out + xi
        out = self.conv(out)

        return out

class ExpandingBlock(nn.Module):
    def __init__(self, ni, residual=False):
        super(ExpandingBlock, self).__init__()

        self.conv_t = nn.ConvTranspose2d(2 * ni, ni, kernel_size=3, stride=2, padding=1, output_padding=1)

        if residual:
            self.conv1 = ResBlock(2 * ni, ni, kernel_size=3, stride=1, padding=1)
            self.conv2 = ResBlock(ni, ni, kernel_size=3, stride=1, padding=1)
            self.conv3 = ResBlock(ni, ni, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = Conv(2 * ni, ni, kernel_size=3, stride=1, padding=1)
            self.conv2 = Conv(ni, ni, kernel_size=3, stride=1, padding=1)
            self.conv3 = Conv(ni, ni, kernel_size=3, stride=1, padding=1)

    def forward(self, pe_out, ff, contr):
        """
            pe_out: Previous Expanding Block output
            ff: adjacent Feed Forward block output
            contr: adjacent Contraction block output
        """
        x1 = x2 = ff
        if pe_out is not None:
            x1 = x2 = self.conv_t(pe_out)
        
        x1 = torch.cat([ff, x1], dim=1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        x = contr + x1 + x2
        x = self.conv3(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, ni, nout, n):
        super(UpSamplingBlock, self).__init__()
        # TODO: Try to use ConvTranspose2d here instead of Upsample.
        self.up = nn.Upsample(scale_factor=(2 ** (n + 2), 2 ** (n + 2)))
        self.conv = Conv(ni, nout, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = self.conv(x)
        return x

class CloudNetp(nn.Module):
    def __init__(
        self,
        n_channels=4,
        n_classes=1,
        inception_depth=6,
        residual=False,
    ):
        super(CloudNetp, self).__init__()

        self.c_blocks = []
        self.f_blocks = []
        self.e_blocks = []
        self.u_blocks = []

        for i in range(0, inception_depth):
            c_block = ContractionBlock(n_channels * (2 ** i), residual)
            self.c_blocks.append(c_block)

        for i in range(1, inception_depth):
            f_block = FeedForwardBlock(n_channels * 2 * (2 ** i), i, residual)
            self.f_blocks.append(f_block)  

        for i in range(1, inception_depth):
            e_block = ExpandingBlock(n_channels * 2  * (2 ** i), residual)
            self.e_blocks.append(e_block)

        for i in range(0, inception_depth - 1):
            u_block = UpSamplingBlock(n_channels * 2 * (2 ** (i + 1)), n_classes, i)
            self.u_blocks.append(u_block)
        u_block = UpSamplingBlock(n_channels * (2 ** inception_depth), n_classes, inception_depth - 2)
        self.u_blocks.append(u_block)
         
        self.out_conv = nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=1)

        self.c_blocks = nn.ModuleList(self.c_blocks)
        self.f_blocks = nn.ModuleList(self.f_blocks)
        self.e_blocks = nn.ModuleList(self.e_blocks)
        self.u_blocks = nn.ModuleList(self.u_blocks)
        
    def forward(self, x: torch.Tensor):

        c_outs = []
        for i in range(len(self.c_blocks)):
            x = self.c_blocks[i](x)
            c_outs.append(x)
        
        f_outs = []
        for i in range(len(self.f_blocks)):
            f_out = self.f_blocks[i](c_outs[:i + 2].copy())
            f_outs.append(f_out)

        e_outs = [0] * len(f_outs)
        e_out = None
        for i in reversed(range(len(self.e_blocks))):
            e_out = self.e_blocks[i](e_out, f_outs[i], c_outs[i + 1])
            e_outs[i] = e_out
        
        n = len(self.u_blocks)
        u_sum = self.u_blocks[n - 1](c_outs[n - 1])
        for i in range(0, n - 1):
            u_sum = u_sum + self.u_blocks[i](e_outs[i])
        
        out = self.out_conv(u_sum)
        return dict(out=out)
