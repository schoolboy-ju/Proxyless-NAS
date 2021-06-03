from torch import nn


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


def get_delta_ops(in_channels, out_channels, kernel_size: tuple, out_h, out_w, num_groups):
    return in_channels * out_channels * kernel_size[0] * kernel_size[1] * out_h * out_w / num_groups


def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = get_delta_ops(
        in_channels=layer.in_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        out_h=out_h,
        out_w=out_w,
        num_groups=layer.groups
    )
    return delta_ops


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self._groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self._groups

        # reshape
        x = x.view(batch_size, self._groups, channels_per_group, height, width)

        # noinspection PyUnresolvedReferences
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batch_size, -1, height, width)
        return x
