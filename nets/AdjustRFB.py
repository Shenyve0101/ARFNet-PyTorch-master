import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class AdaptiveSelectiveBlock(nn.Module):
    def __init__(self, in_planes):
        super(AdaptiveSelectiveBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        )

        self.fc0 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1, x2, x3):
        x = x0 + x1 + x2 + x3
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)

        fusion = self.fc(max_out + avg_out)

        branch_0 = self.sigmoid(self.fc0(fusion))
        branch_1 = self.sigmoid(self.fc1(fusion))
        branch_2 = self.sigmoid(self.fc2(fusion))
        branch_3 = self.sigmoid(self.fc3(fusion))

        return [branch_0, branch_1, branch_2, branch_3]


class AdaptiveSelectiveBlock_13579(nn.Module):
    def __init__(self, in_planes):
        super(AdaptiveSelectiveBlock_13579, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        )

        self.fc0 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc3 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc4 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1, x2, x3, x4):
        x = x0 + x1 + x2 + x3 + x4
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)

        fusion = self.fc(max_out + avg_out)

        branch_0 = self.sigmoid(self.fc0(fusion))
        branch_1 = self.sigmoid(self.fc1(fusion))
        branch_2 = self.sigmoid(self.fc2(fusion))
        branch_3 = self.sigmoid(self.fc3(fusion))
        branch_4 = self.sigmoid(self.fc4(fusion))

        return [branch_0, branch_1, branch_2, branch_3, branch_4]

class AdaptiveSelectiveBlock_135(nn.Module):
    def __init__(self, in_planes):
        super(AdaptiveSelectiveBlock_135, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        )

        self.fc0 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1, x2):
        x = x0 + x1 + x2
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)

        fusion = self.fc(max_out + avg_out)

        branch_0 = self.sigmoid(self.fc0(fusion))
        branch_1 = self.sigmoid(self.fc1(fusion))
        branch_2 = self.sigmoid(self.fc2(fusion))

        return [branch_0, branch_1, branch_2]

class AdaptiveSelectiveBlock_13(nn.Module):
    def __init__(self, in_planes):
        super(AdaptiveSelectiveBlock_13, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        )

        self.fc0 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x0, x1):
        x = x0 + x1
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)

        fusion = self.fc(max_out + avg_out)

        branch_0 = self.sigmoid(self.fc0(fusion))
        branch_1 = self.sigmoid(self.fc1(fusion))

        return [branch_0, branch_1]

class AdaptiveRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(AdaptiveRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0_0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )
        self.branch0_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )

        self.branch1_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
        )
        self.branch1_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, relu=False)
        )

        self.branch2_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 4, kernel_size=3, stride=1, padding=1),
        )
        self.branch2_1 = nn.Sequential(
            BasicConv((inter_planes // 2) * 4, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )

        self.branch3_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 4, kernel_size=3, stride=1, padding=1),
        )
        self.branch3_1 = nn.Sequential(
            BasicConv((inter_planes // 2) * 4, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=4, dilation=4, relu=False)
        )

        self.ASB = AdaptiveSelectiveBlock(2 * inter_planes)

        self.ConvLinear = BasicConv(8 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0_0 = self.branch0_0(x)
        x1_0 = self.branch1_0(x)
        x2_0 = self.branch2_0(x)
        x3_0 = self.branch3_0(x)

        [w0, w1, w2, w3] = self.ASB(x0_0, x1_0, x2_0, x3_0)

        x0 = self.branch0_1(w0 * x0_0)
        x1 = self.branch1_1(w1 * x1_0)
        x2 = self.branch2_1(w2 * x2_0)
        x3 = self.branch3_1(w3 * x3_0)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out

class AdaptiveRFB_13579(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(AdaptiveRFB_13579, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0_0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )
        self.branch0_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )

        self.branch1_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
        )
        self.branch1_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, relu=False)
        )

        self.branch2_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 4, kernel_size=5, stride=1, padding=2),
        )
        self.branch2_1 = nn.Sequential(
            BasicConv((inter_planes // 2) * 4, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )

        self.branch3_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 4, kernel_size=7, stride=1, padding=3),
        )
        self.branch3_1 = nn.Sequential(
            BasicConv((inter_planes // 2) * 4, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=4, dilation=4, relu=False)
        )

        self.branch4_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 4, kernel_size=9, stride=1, padding=4)
        )

        self.branch4_1 = nn.Sequential(
            BasicConv((inter_planes // 2) * 4, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ASB = AdaptiveSelectiveBlock_13579(2 * inter_planes)

        self.ConvLinear = BasicConv(10 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0_0 = self.branch0_0(x)
        x1_0 = self.branch1_0(x)
        x2_0 = self.branch2_0(x)
        x3_0 = self.branch3_0(x)
        x4_0 = self.branch4_0(x)

        [w0, w1, w2, w3, w4] = self.ASB(x0_0, x1_0, x2_0, x3_0, x4_0)

        x0 = self.branch0_1(w0 * x0_0)
        x1 = self.branch1_1(w1 * x1_0)
        x2 = self.branch2_1(w2 * x2_0)
        x3 = self.branch3_1(w3 * x3_0)
        x4 = self.branch4_1(w4 * x4_0)

        out = torch.cat((x0, x1, x2, x3, x4), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out

class AdaptiveRFB_135(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(AdaptiveRFB_135, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0_0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )
        self.branch0_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )

        self.branch1_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
        )
        self.branch1_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, relu=False)
        )

        self.branch2_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 4, kernel_size=3, stride=1, padding=1),
        )
        self.branch2_1 = nn.Sequential(
            BasicConv((inter_planes // 2) * 4, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )


        self.ASB = AdaptiveSelectiveBlock_135(2 * inter_planes)

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0_0 = self.branch0_0(x)
        x1_0 = self.branch1_0(x)
        x2_0 = self.branch2_0(x)

        [w0, w1, w2] = self.ASB(x0_0, x1_0, x2_0)

        x0 = self.branch0_1(w0 * x0_0)
        x1 = self.branch1_1(w1 * x1_0)
        x2 = self.branch2_1(w2 * x2_0)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class AdaptiveRFB_13(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(AdaptiveRFB_13, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0_0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )
        self.branch0_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )

        self.branch1_0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
        )
        self.branch1_1 = nn.Sequential(
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, relu=False)
        )

        self.ASB = AdaptiveSelectiveBlock_13(2 * inter_planes)

        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0_0 = self.branch0_0(x)
        x1_0 = self.branch1_0(x)

        [w0, w1] = self.ASB(x0_0, x1_0)

        x0 = self.branch0_1(w0 * x0_0)
        x1 = self.branch1_1(w1 * x1_0)

        out = torch.cat((x0, x1), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out

if __name__ == '__main__':
    img = torch.rand((1, 128, 256, 256))


    net = AdaptiveRFB_13579(128, 128)
    out = net(img)

    print(out.shape)

    # print('# generator parameters:', sum(param.numel() for param in net.parameters()))
