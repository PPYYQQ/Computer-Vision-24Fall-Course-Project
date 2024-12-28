import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    '''
    创建卷积层，支持不同的下采样模式和填充方式
    '''
    if pad == 'zero':
        padding = kernel_size // 2
        pad_layer = None
    elif pad == 'reflection':
        padding = 0
        pad_layer = nn.ReflectionPad2d(kernel_size // 2)
    else:
        raise NotImplementedError(f"Padding mode '{pad}' is not implemented.")

    layers = []
    if pad_layer is not None:
        layers.append(pad_layer)
    if downsample_mode == 'stride':
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
    elif downsample_mode == 'avg':
        layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias))
    elif downsample_mode == 'max':
        layers.append(nn.MaxPool2d(kernel_size=stride, stride=stride))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias))
    else:
        raise NotImplementedError(f"Downsample mode '{downsample_mode}' is not implemented.")

    return nn.Sequential(*layers)

def bn(num_features):
    '''
    批归一化层
    '''
    return nn.BatchNorm2d(num_features)

def act(act_fun='LeakyReLU'):
    '''
    激活函数
    '''
    if isinstance(act_fun, str):
        if act_fun.lower() == 'leakyrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif act_fun.lower() == 'elu':
            return nn.ELU(inplace=True)
        else:
            return nn.Identity()
    else:
        return act_fun()

class Concat(nn.Module):
    '''
    连接多个子模块的输出
    '''
    def __init__(self, dim, *modules):
        super(Concat, self).__init__()
        self.dim = dim
        self.modules_list = nn.ModuleList(modules)

    def forward(self, input):
        outputs = []
        for module in self.modules_list:
            outputs.append(module(input))
        return torch.cat(outputs, dim=self.dim)

def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """
    构建带有跳跃连接的编码器-解码器网络。

    参数：
        act_fun: 激活函数，可以是 'LeakyReLU'、'ELU'、'ReLU' 或自定义的 nn.Module
        pad (str): 填充模式 'zero' 或 'reflection'，默认 'zero'
        upsample_mode (str): 上采样模式 'nearest' 或 'bilinear'，默认 'nearest'
        downsample_mode (str): 下采样模式 'stride', 'avg', 或 'max'，默认 'stride'

    返回：
        nn.Sequential: 定义的网络模型
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(n_scales):

        deeper = nn.Sequential()
        skip_layer = nn.Sequential()

        if num_channels_skip[i] != 0:
            concat = Concat(1, skip_layer, deeper)
            model_tmp.add_module(f"concat_{i}", concat)
        else:
            model_tmp.add_module(f"deeper_{i}", deeper)
        
        out_channels = num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
        model_tmp.add_module(f"bn_{i}", bn(out_channels))

        if num_channels_skip[i] != 0:
            skip_layer.add_module(f"conv_skip_{i}", conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip_layer.add_module(f"bn_skip_{i}", bn(num_channels_skip[i]))
            skip_layer.add_module(f"act_skip_{i}", act(act_fun))
        
        deeper.add_module(f"conv_down_{i}", conv(input_depth, num_channels_down[i], filter_size_down[i], stride=2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module(f"bn_down_{i}", bn(num_channels_down[i]))
        deeper.add_module(f"act_down_{i}", act(act_fun))

        deeper.add_module(f"conv_down2_{i}", conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add_module(f"bn_down2_{i}", bn(num_channels_down[i]))
        deeper.add_module(f"act_down2_{i}", act(act_fun))

        deeper_main = nn.Sequential()

        if i == last_scale:
            k = num_channels_down[i]
        else:
            deeper.add_module(f"deeper_main_{i}", deeper_main)
            k = num_channels_up[i + 1]

        deeper.add_module(f"upsample_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add_module(f"conv_up_{i}", conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], stride=1, bias=need_bias, pad=pad))
        model_tmp.add_module(f"bn_up_{i}", bn(num_channels_up[i]))
        model_tmp.add_module(f"act_up_{i}", act(act_fun))

        if need1x1_up:
            model_tmp.add_module(f"conv1x1_up_{i}", conv(num_channels_up[i], num_channels_up[i], 1, stride=1, bias=need_bias, pad=pad))
            model_tmp.add_module(f"bn1x1_up_{i}", bn(num_channels_up[i]))
            model_tmp.add_module(f"act1x1_up_{i}", act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add_module("final_conv", conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add_module("sigmoid", nn.Sigmoid())

    return model
