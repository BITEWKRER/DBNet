# -*-coding:utf-8 -*-
"""
# Time       ：2023/10/28 13:47
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from torch import nn
from torch.nn import Linear, Dropout

from utils.norm import Swish, unetconv3d, channel_shuffle, channel_spilt, Softmax, conv3d


class Attention(nn.Module):
    def __init__(self, channel, norm=nn.BatchNorm3d):
        super(Attention, self).__init__()

        self.channel = channel
        self.softmax = Softmax()  # nn.Softmax(dim=-1)  #

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.active = norm(channel)


class mScale(Attention):
    def __init__(self, in_channels=1, hwd=32, block=unetconv3d, dilation=False, norm=nn.BatchNorm3d):
        super(mScale, self).__init__(in_channels)

        self.layer1 = block(in_channels, in_channels, n=1, ks=(3, 3, hwd), padding=(1, 1, 0))
        self.layer2 = block(in_channels, in_channels, n=1, ks=(3, hwd, 3), padding=(1, 0, 1))
        self.layer3 = block(in_channels, in_channels, n=1, ks=(hwd, 3, 3), padding=(0, 1, 1))
        self.active = norm(in_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        zaxis = self.layer1(x)
        yaxis = self.layer2(x)
        xaxis = self.layer3(x)
        spx = self.active(zaxis + yaxis + xaxis) + x
        return spx

class DenseCA(Attention):
    def __init__(self, inchannel, shufinle=8, reduction=2, block=unetconv3d, norm=nn.BatchNorm3d):
        super(DenseCA, self).__init__(inchannel)
        self.inchannel = inchannel
        self.shufinle = shufinle

        self.catfusion = block(inchannel * 2, inchannel, n=1, ks=1, padding=0)

        self.se1 = nn.Sequential(
            block(inchannel, inchannel // reduction, n=1, ks=1, padding=0),
            block(inchannel // reduction, inchannel, n=1, ks=1, padding=0)
        )

        self.se2 = nn.Sequential(
            block(inchannel, inchannel // (reduction * 2), n=1, ks=1, padding=0),
            block(inchannel // (reduction * 2), inchannel, n=1, ks=1, padding=0)
        )

        self.se3 = nn.Sequential(
            block(inchannel, inchannel // (reduction * 4), n=1, ks=1, padding=0),
            block(inchannel // (reduction * 4), inchannel, n=1, ks=1, padding=0)
        )

        self.se4 = nn.Sequential(
            block(inchannel, inchannel // (reduction * 8), n=1, ks=1, padding=0),
            block(inchannel // (reduction * 8), inchannel, n=1, ks=1, padding=0)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgv = self.avg_pool(x)
        maxv = self.max_pool(x)
        hx = self.catfusion(torch.cat((avgv, maxv), dim=1))

        sx = channel_shuffle(hx, self.shufinle)

        exp1 = self.se1(hx)
        exp2 = self.se2(exp1) + exp1
        exp3 = self.se3(exp2) + exp2 + exp1
        exp = self.se4(exp3) + exp2 + exp1 + exp3

        return self.sigmoid((sx + exp + hx) * x)  # self.sigmoid()


class SP(nn.Module):

    def __init__(self, in_channels, hwd=32, norm=nn.BatchNorm3d):
        super(SP, self).__init__()

        self.sp = DenseSP(in_channels, hwd=hwd, block=conv3d)
        # self.sp = mScale(in_channels, hwd, block=conv3d)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        xsp = self.sp(x) * x

        x1a, x2a = channel_spilt(xsp)

        return x1a, x2a

class CA(nn.Module):

    def __init__(self, in_channels, depth=1, norm=nn.BatchNorm3d):
        super(CA, self).__init__()

        self.ca = DenseCA(in_channels, depth, block=conv3d)
        # self.cbs1 = conv3d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)
        # self.cbs2 = conv3d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)
        # self.sp = SpatialAttention()
        rate = 32
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm3d(int(in_channels / rate)),
            Swish(),
            nn.Conv3d(int(in_channels / rate), in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        xsp = self.ca(x) * x
        x1a, x2a = channel_spilt(xsp)
        return x1a, x2a


class CASP(nn.Module):

    def __init__(self, in_channels, depth=1, hwd=32, norm=nn.BatchNorm3d):
        super(CASP, self).__init__()

        rate = depth
        # 层
        self.sp = nn.Sequential(
            nn.Conv3d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm3d(int(in_channels / rate)),
            mScale(int(in_channels / rate), hwd, block=conv3d),
            nn.Conv3d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm3d(in_channels)
        )
        self.ca = DenseCA(in_channels, depth, block=conv3d)

    def forward(self, x1, x2):
        hx = torch.cat((x1, x2), dim=1)

        xc1 = self.ca(hx) * hx
        xb1 = self.sp(xc1).sigmoid() * hx
        xa, xb = channel_spilt(xb1)

        return xa, xb
