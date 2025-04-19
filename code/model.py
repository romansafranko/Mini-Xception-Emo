"""
CNN backbone based on the Mini‑Xception architecture with depth‑wise
separable convolutions, Squeeze‑and‑Excitation blocks and residual links.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Channel‑wise attention (Hu et al., 2018)."""

    def __init__(self, channels, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """
    Factorised convolution: depth‑wise spatial conv followed by a 1×1 point‑wise
    conv. Greatly reduces parameter count and FLOPs.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    """Two depth‑wise separable convs + SE attention + optional down‑sample."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

        self.se = SEBlock(out_channels, reduction=16)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out += identity
        return self.relu(out)


class MiniXception(nn.Module):
    """Compact Xception variant for 128×128 gray images."""

    def __init__(self, num_classes: int = 7, in_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stacked residual blocks (first four down‑sample by stride = 2)
        self.block1 = ResidualBlock(64, 128, stride=2)
        self.block2 = ResidualBlock(128, 256, stride=2)
        self.block3 = ResidualBlock(256, 512, stride=2)
        self.block4 = ResidualBlock(512, 512, stride=2)
        self.block5 = ResidualBlock(512, 512, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.dropout(self.block1(x))
        x = self.dropout(self.block2(x))
        x = self.dropout(self.block3(x))
        x = self.dropout(self.block4(x))
        x = self.dropout(self.block5(x))

        x = self.pool(x).flatten(1)
        x = self.fc_dropout(x)
        return self.fc(x)
