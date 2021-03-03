import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activat = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activat(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels // 2, out_channels // 2, 3)
        self.conv3 = ConvBlock(out_channels // 2, out_channels // 2, 3)
        self.conv4 = ConvBlock(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        x = self.conv1(x)
        feat = x
        c = self.out_channels
        x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        feat1 = x
        x = self.conv3(x)
        x = torch.cat([x, feat1], dim=1)
        x = self.conv4(x)
        feat1 = x
        x = torch.cat([feat, x], dim=1)
        feat2 = self.maxpool(x)

        return feat2, feat1


class CSPDarkNet53(nn.Module):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.resblock_body1 = ResBlock(64, 64)
        self.resblock_body2 = ResBlock(128, 128)
        self.resblock_body3 = ResBlock(256, 256)
        self.conv3 = ConvBlock(512, 512, kernel_size=3)
        # self.num_features = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.resblock_body1(x)
        x, _ = self.resblock_body2(x)
        x, feat1 = self.resblock_body3(x)
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


def CSPDarknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet53()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
