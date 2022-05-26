import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ChannelFusionFactor(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(ChannelFusionFactor, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(True),
            nn.Sigmoid()
        )

    def  forward(self, x):
        b, c, h, w = x.size()
        out = self.conv(x)
        out = self.relu(self.bn(out))
        out = self.fc(self.avg_pool(out).view(b, c))
        y = out.view(b, c, 1, 1)
        return y
        # return x * y.expand_as(x)

class PositionFusionFactor(nn.Module):
    def __init__(self, in_channel):
        super(PositionFusionFactor, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.relu(self.bn(self.conv(x)))

        y = torch.mean(y, dim=1, keepdim=True)
        y = self.sigmoid(y)
        return y
        # return x * y.expand_as(x)


class DFF(nn.Module):
    def __init__(self, level, in_channels, dim):
        super(DFF, self).__init__()

        self.level = level
        self.dim = dim

        if self.level == 0:
            self.level_1_0 = add_conv(in_channels, self.dim, 3, 1)
            self.level_2_0 = add_conv(in_channels, self.dim, 3, 1)
            self.expand = add_conv(self.dim, self.dim, 3, 1)
        elif self.level == 1:
            self.level_0_1 = add_conv(in_channels, self.dim, 3, 2)
            self.level_2_1 = add_conv(in_channels, self.dim, 3, 1)
            self.expand = add_conv(self.dim, self.dim, 3, 1)
        elif self.level == 2:
            self.level_0_2_1 = add_conv(in_channels, self.dim, 3, 2)
            self.level_0_2_2 = add_conv(self.dim, self.dim, 3, 2)
            self.level_1_2 = add_conv(in_channels, self.dim, 3, 2)
            self.expand = add_conv(self.dim, self.dim, 3, 1)

        self.conv0 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, P3, P4, P5):
        P3_w, P3_h = P3.size()[2:]
        P4_w, P4_h = P4.size()[2:]
        P5_w, P5_h = P5.size()[2:]

        if self.level == 0:
            level0 = P3
            c4_resized = F.interpolate(P4, size=(P3_w, P3_h))
            c5_resized = F.interpolate(P5, size=(P3_w, P3_h))
            level1 = self.level_1_0(c4_resized)
            level2 = self.level_2_0(c5_resized)

        elif self.level == 1:
            level1 = P4
            level0 = self.level_0_1(P3)
            c5_resized = F.interpolate(P5, size=(P4_w, P4_h))
            level2 = self.level_2_1(c5_resized)
        elif self.level == 2:
            level2 = P5
            level0 = self.level_0_2_1(P3)
            level0 = self.level_0_2_2(level0)
            level1 = self.level_1_2(P4)

        level_all = self.conv0(level0 + level1 + level2)

        avg_out = torch.mean(level_all, dim=1, keepdim=True)
        max_out, _ = torch.max(level_all, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':

    x1 = torch.rand((1, 256, 64, 64))
    x2 = torch.rand((1, 256, 32, 32))
    x3 = torch.rand((1, 256, 16, 16))

    model = DFF(0, 256, 256)
    out = model(x1, x2 ,x3)










