import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])

def std(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)

def default_conv_stride2(in_channels, out_channels, kernel_size, stride=2, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=1, bias=bias)

def default_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class CEALayer(nn.Module):
    def __init__(self, n_feats=64, reduction=16):
        super(CEALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 5, padding=1, groups=n_feats // reduction, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return y


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class UpsampleBlock(nn.Module):
    def __init__(self, scale=4, n_feats=64, kernel_size=3, stride=1, bias=True, bn=False, act=nn.ReLU(True), conv_=default_conv):
        super(UpsampleBlock, self).__init__()

        self.op = []
        if (scale & (scale - 1)) == 0:
            for i in range(int(math.log2(scale))):
                self.op.append(conv_(n_feats, 4 * n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
                self.op.append(nn.PixelShuffle(2))
                if bn:
                    self.op.append(nn.BatchNorm2d(n_feats))
                if act is not None:
                    self.op.append(act)
        elif(scale == 3):
            self.op.append(conv_(n_feats, 9 * n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
            self.op.append(nn.PixelShuffle(3))
            if bn:
                self.op.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                self.op.append(act)
        else:
            raise NotImplementedError

        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        x = self.op(x)
        return x