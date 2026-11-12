"""OSNet-AIN x1.0, a person re-identification network (feature extractor only).

Ported from Torchreid ``torchreid/models/osnet_ain.py`` (Kaiyang Zhou,
https://github.com/KaiyangZhou/deep-person-reid), MIT License:

    Copyright (c) 2018 Kaiyang Zhou

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Changes from upstream: the classifier head, the loss branches, the other widths
and the ImageNet-init code are removed; module names are unchanged so the
upstream state dict (minus ``classifier.*``) loads with ``strict=True``. In eval
mode :class:`OSNetAIN` returns the 512-d ``fc`` feature (Linear + BatchNorm1d +
ReLU of the pooled map), as upstream does.

Paper: Zhou et al., "Learning Generalisable Omni-Scale Representations for
Person Re-Identification", TPAMI 2021 (arXiv:1910.06827).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvLayer(nn.Module):
    """Convolution + BatchNorm (or affine InstanceNorm) + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 convolution + BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + optional BatchNorm, no activation."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution: 1x1 (linear) + depth-wise 3x3, then BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    """``depth`` stacked :class:`LightConv3x3` (one receptive-field scale)."""

    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        layers = [LightConv3x3(in_channels, out_channels)]
        layers += [LightConv3x3(out_channels, out_channels) for _ in range(depth - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    """Aggregation gate shared by the streams of a block (sigmoid channel weights)."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        w = self.global_avgpool(x)
        w = self.gate_activation(self.fc2(self.relu(self.fc1(w))))
        return x * w


class OSBlock(nn.Module):
    """Omni-scale block: four streams (1 to 4 light 3x3 convs) fused by the gate, plus a residual."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super().__init__()
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList([LightConvStream(mid_channels, mid_channels, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def _streams(self, x):
        x1 = self.conv1(x)
        x2 = 0
        for stream in self.conv2:
            x2 = x2 + self.gate(stream(x1))
        return x2

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.conv3(self._streams(x)) + identity)


class OSBlockINin(OSBlock):
    """:class:`OSBlock` with instance normalisation inside the residual branch (the "AIN" blocks)."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4):
        super().__init__(in_channels, out_channels, reduction, T)
        self.conv3 = Conv1x1Linear(out_channels // reduction, out_channels, bn=False)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.IN(self.conv3(self._streams(x))) + identity)


class OSNetAIN(nn.Module):
    """OSNet-AIN x1.0 backbone + 512-d embedding layer (no classifier)."""

    blocks = ((OSBlockINin, OSBlockINin), (OSBlock, OSBlockINin), (OSBlockINin, OSBlock))
    channels = (64, 256, 384, 512)
    feature_dim = 512

    def __init__(self):
        super().__init__()
        c = self.channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(self.blocks[0], c[0], c[1])
        self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._make_layer(self.blocks[1], c[1], c[2])
        self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._make_layer(self.blocks[2], c[2], c[3])
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], self.feature_dim), nn.BatchNorm1d(self.feature_dim), nn.ReLU())

    @staticmethod
    def _make_layer(blocks, in_channels, out_channels):
        layers = [blocks[0](in_channels, out_channels)]
        layers += [block(out_channels, out_channels) for block in blocks[1:]]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        return self.fc(self.global_avgpool(x).flatten(1))


def load_osnet_ain(path) -> OSNetAIN:
    """OSNet-AIN x1.0 in eval mode with the weights at ``path`` (strict: every tensor must match)."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith('classifier.')}
    model = OSNetAIN()
    model.load_state_dict(state, strict=True)
    return model.eval()
