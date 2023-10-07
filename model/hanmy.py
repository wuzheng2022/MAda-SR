import torch
import torch.nn as nn
from model.common import UpsampleBlock, default_conv


def make_model(args, parent=False):
    return HAN(args)


class ChannelAttentation(nn.Module):
    def __init__(self, channels=64, reduction=16, conv_=default_conv):
        super(ChannelAttentation, self).__init__()

        self.op = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            conv_(in_channels=channels, out_channels=channels // reduction, kernel_size=1, stride=1, padding=0,
                  bias=True),
            nn.ReLU(inplace=True),
            conv_(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1, padding=0,
                  bias=True),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        s = self.op(x)
        x = x * s
        return x

class ResChannelAttBlock(nn.Module):
    def __init__(self, n_feats=64, reduction=16, kernel_size=3, stride=1, bias=True, bn=False, instance_norm=False,
                 act=nn.ReLU(True), conv_=default_conv):
        super(ResChannelAttBlock, self).__init__()
        assert act is not None
        self.op = []
        for i in range(2):
            self.op.append(conv_(n_feats, n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
            if bn:
                self.op.append(nn.BatchNorm2d(n_feats))
            if instance_norm:
                self.op.append(nn.InstanceNorm2d(n_feats))
            if i == 0:
                self.op.append(act)
        self.op.append(ChannelAttentation(channels=n_feats, reduction=reduction))
        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        res = self.op(x)
        x = res + x
        return x


class ResGroup(nn.Module):
    def __init__(self, n_rcab=20, n_feats=64, reduction=16, kernel_size=3, stride=1, bias=True, bn=False,
                 instance_norm=False, act=nn.ReLU(True), conv_=default_conv):
        super(ResGroup, self).__init__()
        assert act is not None
        self.op = []

        for _ in range(n_rcab):
            self.op.append(ResChannelAttBlock(n_feats=n_feats, reduction=reduction, kernel_size=kernel_size,
                                              stride=stride, bias=bias, bn=bn, instance_norm=instance_norm, act=act))

        self.op.append(conv_(in_channels=n_feats, out_channels=n_feats, kernel_size=kernel_size, stride=stride, bias=bias))
        self.op = nn.Sequential(*self.op)

    def forward(self, x):
        res = self.op(x)
        x = res + x
        return x


class LAM(nn.Module):
    def __init__(self):
        super(LAM, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        bs, N, C, H, W = x.size()
        query = x.view(bs, N, -1)
        key = x.view(bs, N, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        attention = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(attention)

        value = x.view(bs, N, -1)
        out = torch.bmm(attention, value)
        out = out.view(bs, N, C, H, W)

        x = self.gamma * out + x
        x = x.view(bs, -1, H, W)
        return x

class CSAM(nn.Module):
    def __init__(self):
        super(CSAM, self).__init__()

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        # out = x.unsqueeze(1)
        # out = self.sigmoid(self.conv(out))

        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) # CxC
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)

        out = self.gamma * out
        out = out.view(m_batchsize, C, height, width)
        x = x * out + x
        return x

class HAN(nn.Module):
    def __init__(self, args, in_c=3, out_c=3, act=nn.ReLU(True), global_res=True, conv=default_conv):
        super(HAN, self).__init__()
        self.global_res = global_res
        kernel_size = 3
        self.scale = args.scale[0]
        n_feats = 128
        n_rg = 10
        n_rcab = 20
        self.head = conv(in_c, n_feats, kernel_size)

        self.body = nn.ModuleList()
        for _ in range(n_rg):
            self.body.append(
                ResGroup(n_rcab=n_rcab, n_feats=n_feats, kernel_size=3, stride=1, bias=True, act=act)
            )

        self.csam = CSAM()
        self.lam = LAM()
        self.last_conv = conv((n_rg+1) * n_feats, n_feats, kernel_size)

        self.tail = nn.Sequential(*[
            UpsampleBlock(scale=self.scale, n_feats=n_feats, kernel_size=3, stride=1, bias=True, bn=False, act=act),
            conv(n_feats, out_c, kernel_size)]
          )

    def forward(self, x):
        if self.global_res:
            x0 = nn.Upsample(scale_factor=self.scale, mode='bicubic')(x)
        x = self.head(x)

        res = x
        res1 = []
        for i in range(len(self.body)):
            res = self.body[i](res)
            res1.append(res.unsqueeze(1))

        res1 = torch.cat(res1, dim=1)
        la = self.lam(res1)
        csa = self.csam(res)

        res = torch.cat([la, csa], dim=1)
        res = self.last_conv(res)

        x = x + res
        x = self.tail(x)
        if self.global_res:
            x = x + x0
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))