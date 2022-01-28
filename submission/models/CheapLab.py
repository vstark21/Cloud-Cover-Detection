# The MIT License (MIT)
# =====================
#
# Copyright © 2019-2020 Azavea
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn

class LearnedIndices(nn.Module):

    output_channels = 32

    def __init__(self, band_count):
        super(LearnedIndices, self).__init__()
        intermediate_channels1 = 64
        kernel_size = 1
        padding_size = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(band_count,
                                     intermediate_channels1,
                                     kernel_size=kernel_size,
                                     padding=padding_size,
                                     bias=False)
        self.conv_numerator = nn.Conv2d(intermediate_channels1,
                                              self.output_channels,
                                              kernel_size=1,
                                              padding=0,
                                              bias=False)
        self.conv_denominator = nn.Conv2d(intermediate_channels1,
                                                self.output_channels,
                                                kernel_size=1,
                                                padding=0,
                                                bias=True)
        self.batch_norm_quotient = nn.BatchNorm2d(self.output_channels)

    def forward(self, x):
        x = self.conv1(x)
        numerator = self.conv_numerator(x)
        denomenator = self.conv_denominator(x)
        x = numerator / (denomenator + 1e-7)
        x = self.batch_norm_quotient(x)
        return x


class Nugget(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(Nugget, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class CheapLab(nn.Module):
    def __init__(self,
                 num_channels: int,
                 out_channels: int = 2):
        super(CheapLab, self).__init__()

        self.indices = LearnedIndices(num_channels)
        self.classifier = nn.Sequential(
            Nugget(1, self.indices.output_channels + num_channels, 16),
            Nugget(1, 16, 8),
            Nugget(1, 8, 4),
            Nugget(1, 4, out_channels),
        )
        self.out_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor):
        x = torch.cat([self.indices(x), x], axis=1)
        x = self.classifier(x)
        x = self.out_conv(x)
        return dict(out=x)
