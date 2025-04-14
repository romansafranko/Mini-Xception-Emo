import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.se = SEBlock(out_channels, reduction=16)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out

class MiniXception(nn.Module):
    def __init__(self, num_classes=7, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.block1 = ResidualBlock(64, 128, stride=2)
        self.block2 = ResidualBlock(128, 256, stride=2)
        self.block3 = ResidualBlock(256, 512, stride=2)
        self.block4 = ResidualBlock(512, 512, stride=2)
        self.block5 = ResidualBlock(512, 512, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.pool    = nn.AdaptiveAvgPool2d((1,1))

        self.fc_dropout = nn.Dropout(0.3)
        self.fc         = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = self.block3(x)
        x = self.dropout(x)

        x = self.block4(x)
        x = self.dropout(x)

        x = self.block5(x)
        x = self.dropout(x)

        x = self.pool(x)
        x = self.fc_dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
