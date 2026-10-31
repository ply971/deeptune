"""A tiny, untrained ResNet (``--model_version resnet_tiny``) for tests and CPU smoke runs. Never pretrained."""
import torch.nn as nn
from torchvision.models.resnet import BasicBlock


class TinyResNet(nn.Module):
    """The torchvision ResNet layout (conv1, bn1, relu, maxpool, layer1-4, avgpool, fc) with one narrow
    BasicBlock per stage and a total stride of 16.

    Unlike a real ResNet the pool keeps a 2x2 grid, so the feature says where things are. The grid was not shown to help
    cnn_rnn on the direction benchmark. Oracle passes over 5 model seeds on one dataset: from 240 clips 1/5 with this grid
    and 0/5 with a global pool; from 800 clips 4/5 with this grid and 5/5 with a global pool. Five seeds cannot separate
    these. See docs/video-plan/reports/B2.md."""

    GRID = 2

    def __init__(self, widths=(8, 16, 32, 64)):
        super().__init__()
        self.inplanes = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.layer1 = self._stage(widths[0], stride=1)
        self.layer2 = self._stage(widths[1], stride=2)
        self.layer3 = self._stage(widths[2], stride=2)
        self.layer4 = self._stage(widths[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(self.GRID)
        self.fc = nn.Linear(widths[3] * self.GRID ** 2, 1)

    def _stage(self, planes, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        block = BasicBlock(self.inplanes, planes, stride, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))
