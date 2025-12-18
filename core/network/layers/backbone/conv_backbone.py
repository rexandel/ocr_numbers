import torch.nn as nn
from enum import Enum
from typing import List


class _residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, activation):
        super(_residual_block, self).__init__()
        self.skip = nn.Sequential()
        stride = 1
        if in_ch != out_ch:
            stride = 2
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class block_type(Enum):
    basic = _residual_block


class conv_backbone(nn.Module):
    def __init__(self, in_channels, out_channels, block, layer_sizes):
        super(conv_backbone, self).__init__()
        act = nn.ReLU(inplace=True)
        self.layer_sizes = layer_sizes
        ch = int(out_channels / (2 ** (len(self.layer_sizes) + 1)))
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            act,
            nn.Conv2d(ch, ch * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            act,
        )
        ch *= 2
        self.stages = nn.ModuleList()
        for num_blocks in self.layer_sizes:
            self.stages.append(self._create_stage(ch, ch * 2, block, num_blocks, act))
            ch *= 2

    @staticmethod
    def _create_stage(in_ch, out_ch, block, num_blocks, act):
        layers = [block.value(in_ch, out_ch, act)]
        for _ in range(num_blocks - 1):
            layers.append(block.value(out_ch, out_ch, act))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

