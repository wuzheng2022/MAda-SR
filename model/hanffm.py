from model import common
import torch
import torch.nn as nn
import pdb

def make_model(args, parent=False):
    return HANFFM(args)


class FFM_Module(nn.Module):
    def __init__(self, in_dim):
        super(FFM_Module, self).__init__()
        self.channel_in = in_dim
        
        self.convw = nn.Conv3d(1, 1, 3, 1, 1)
        self.conv_dr = nn.Sequential(
                nn.Conv2d(self.channel_in*2, self.channel_in, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
        )
        self.conv_dw = nn.Sequential(
                nn.Conv2d(self.channel_in, self.channel_in, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
        )
        self.conv_dwr = nn.Sequential(
                nn.Conv2d(self.channel_in*2, self.channel_in, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x0, x1, x2):
        m_batchsize, C, height, width = x1.size()
        # 1
        out1 = x1.unsqueeze(1)
        out1 = self.sigmoid(self.convw(out1))
        out1 = self.gamma*out1
        out1 = out1.view(m_batchsize, -1, height, width)
        
        # 2
        out2 = self.conv_dr(torch.cat([x0, x2], 1))
        out2 = out1 * out2 + out2
        
        # 3
        out = self.conv_dw(out2)
    
        # 4
        out = self.conv_dwr(torch.cat([out, x0],1))
        x0 = x0 + out
        return x0
        

class MH_Module(nn.Module):
    def __init__(self, in_dim):
        super(MH_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv1 = nn.Conv3d(1, 1, 1, 1, 0)
        self.conv3 = nn.Conv3d(1, 1, 3, 1, 1)
        self.conv5 = nn.Conv3d(1, 1, 5, 1, 2)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        
        out1 = self.conv1(out)
        out2 = self.conv3(out)
        out3 = self.conv5(out)
        
        # print(out1.size(), out2.size(), out3.size())
        # reshape
        proj_query = out1.view(m_batchsize, 1, -1)
        proj_key = out2.view(m_batchsize, 1, -1).permute(0, 2, 1)
        # 
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = out3.view(m_batchsize, 1, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, 1, C, height, width)
        out = self.sigmoid(out)
        
        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x
    
    
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

    
class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Holistic Attention Network (HAN)
class HANFFM(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HANFFM, self).__init__()
        self.global_res = True
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        self.scale = args.scale[0]
        self.reg_lambda = args.reg_lambda
        self.reg_params = None
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = MH_Module(n_feats) #CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

        # depth-width fusion module
        self.fusion = FFM_Module(n_feats)
        # self.mylast = nn.Conv2d(n_feats, n_feats, 3, 1, 1)


    def forward(self, x):
        if self.global_res:
            x0 = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=True)(x)

        x = self.head(x)
        res = x
        #pdb.set_trace()
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            #print(name)
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        
        out1 = res
        
        res = self.la(res1)
        out2 = self.last_conv(res)
        out1 = self.csa(out1)

        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        
        # 加入cas和la的融合模块
        # 测试改进模块的作用
        fout = self.fusion(x, out1, out2)
        res = 0.1 * fout + res 

        res += x

        x = self.tail(res)
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
        
    def init_reg_params(self):
        reg_params = {}

        for name, param in self.named_parameters():

            # print("Initializing omega values for layer", name)
            omega = torch.zeros(param.size())

            init_val = param.data.clone()
            param_dict = {}

            # for first task, omega is initialized to zero
            param_dict['omega'] = omega
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

        self.reg_params = reg_params

    def init_reg_params_across_tasks(self):
        reg_params = self.reg_params

        for name, param in self.named_parameters():
            # print("Initializing omega values for layer for the new task", name)
            # 获取该参数的字典
            param_dict = reg_params[param]

            # Store the previous values of omega
            prev_omega = param_dict['omega']

            # Initialize a new omega
            new_omega = torch.zeros(param.size())

            init_val = param.data.clone()

            # 多加了一个前Ω的属性
            param_dict['prev_omega'] = prev_omega
            # 再把当前任务的Ω初始化为0
            param_dict['omega'] = new_omega

            # 存储参数在前一个任务训练后保留下来的值，作为该任务的初始值
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

        self.reg_params = reg_params

    @property
    def name(self):
        return (
            'HANFFM'
            '-reg_lambda{reg_lambda}'
        ).format(
            reg_lambda=self.reg_lambda,
        )

    def init_params(self):
        params = []
        for param in self.parameters():
            params.append(param.detach())
        return params

    def get_params(self):
        params = []
        for param in self.parameters():
            params.append(param)
        return params