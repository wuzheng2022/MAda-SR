from re import sub
from turtle import st
from model import common
import torch
import torch.nn as nn
import pdb

def make_model(args, parent=False):
    return CSN(args)


class CSB(nn.Module):
    def __init__(self, num_maps, num_stage=4, growth=64, conv=common.default_conv) -> None:
        super(CSB, self).__init__()
        self.num_stage = num_stage
        sub_maps = num_maps // 2
        self.res_branch = nn.Sequential(
            conv(sub_maps, sub_maps, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(sub_maps, sub_maps, kernel_size=3)
        )
        self.dense_branch = nn.Sequential(
            conv(sub_maps, growth, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.dense = conv(sub_maps+growth, sub_maps, kernel_size=3)
        self.merge = conv(num_maps, num_maps, kernel_size=1)

    def forward(self, x):
        
        # split by channels
        splits0, splits1 = torch.split(x, x.size(1)//2, dim=1)

        for i in range(self.num_stage):
            
            mean_tensor = (splits0 + splits1) / 2

            splits0 = self.res_branch(splits0)
            splits0 += mean_tensor

            cur_input = self.dense_branch(splits1)
            cur_input = torch.concat([splits1, cur_input], dim=1)
            splits1 = self.dense(cur_input)
            splits1 += mean_tensor
        
        res = torch.concat([splits0, splits1], dim=1)
        res = self.merge(res)
        res = 0.1 * res + x
        return res


class CSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CSN, self).__init__()
        n_feats = 256 if args.n_feats is None else args.n_feats
        n_resgroups = 4 if args.n_resblocks is None else args.n_resblocks
        n_colors = args.n_colors
        self.scale = args.scale[0]

        self.FEN = nn.Sequential(
            conv(n_colors, n_feats, kernel_size=3),
            conv(n_feats, n_feats, kernel_size=1),
            conv(n_feats, n_feats, kernel_size=3),
        )
        # define body module
        modules_body = [
            CSB(n_feats) for _ in range(n_resgroups)]
        

        self.NMN = nn.Sequential(*modules_body)

        self.global_merge = nn.Sequential(
            conv(n_feats*n_resgroups, n_feats, kernel_size=1),
            conv(n_feats, n_feats, kernel_size=3)
        )
        self.tail = nn.Sequential(
            common.Upsampler(conv, self.scale, n_feats),
            conv(n_feats, n_colors, kernel_size=1)
        )

    def forward(self, x):
        external_shortcut = nn.Upsample(scale_factor=self.scale, mode='bicubic')(x)

        res = self.FEN(x)

        global_shortcut = res
        for name, midlayer in self.NMN._modules.items():
            res = midlayer(res)
            if name == '0':
                res1 = res
            else:
                res1 = torch.cat([res, res1], 1)

        res = self.global_merge(res1)
        res += global_shortcut

        res = self.tail(res)
        res += external_shortcut

        # a = torch.zeros(1).to(torch.device('cuda'))
        # b = torch.ones(1).to(torch.device('cuda'))
        # res = torch.minimum(torch.maximum(res, a), b)

        return res


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