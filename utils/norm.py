# -*-coding:utf-8 -*-
"""
# Time       ：2023/4/12 18:00
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from torch import nn
from torch.autograd import Variable


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class LogSoftmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_max, _ = x.max(dim=self.dim, keepdim=True)  # 数值平移，避免溢出
        return x - x_max - torch.log(torch.sum(torch.exp(x - x_max), dim=self.dim, keepdim=True))


class Softmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim
        self.log_softmax = LogSoftmax(dim=self.dim)

    def forward(self, x):
        return torch.exp(self.log_softmax(x))


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width, z = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width, z)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width, z)
    return x


def channel_spilt(x):
    x1, x2 = x.chunk(2, dim=1)
    return x1, x2

class conv3d(nn.Module):
    def __init__(self, in_size, out_size, n=2, ks=3, s=1, padding=1, norm=nn.BatchNorm3d):
        super(conv3d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = s
        self.padding = padding
        p = padding
        if n == 1:
            # self.block = resnetblock(in_size, out_size, n=n, ks=ks, s=s, padding=padding)
            self.block = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p, bias=False),
                                       norm(out_size),
                                       Swish())
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p, bias=False),
                                     norm(out_size),
                                     Swish())
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        if self.n == 1:
            return self.block(inputs)
        else:
            x = inputs
            for i in range(1, self.n + 1):
                conv = getattr(self, 'conv%d' % i)
                x = conv(x)

            return x

class unetconv3d(nn.Module):
    def __init__(self, in_size, out_size, n=2, ks=3, s=1, padding=1, norm=nn.BatchNorm3d):
        super(unetconv3d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = s
        self.padding = padding
        p = padding

        if n == 1:
            self.cb = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=ks, stride=s, padding=p, bias=False),
                norm(out_size)
            )
            self.swish = Swish()
            self.res = nn.Conv3d(in_size, out_size, kernel_size=3, padding=1, bias=False) # 改成了3
        else:
            self.cb = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=ks, stride=s, padding=p, bias=False),
                norm(out_size),
                Swish(),
                nn.Conv3d(out_size, out_size, kernel_size=ks, stride=s, padding=p, bias=False),
                norm(out_size)
            )
            self.swish = Swish()

            self.res = nn.Conv3d(in_size, out_size, kernel_size=1, padding=0, bias=False)

    def forward(self, inputs):
        cb = self.cb(inputs)
        x = self.res(inputs)
        return self.swish(cb + x)


if __name__ == '__main__':
    var = torch.rand(1, 32, 64, 64, 64)
    x = Variable(var).cuda()
    model = acres(32, 64, 2).cuda()
    x = model(x)
    print(x.size())
