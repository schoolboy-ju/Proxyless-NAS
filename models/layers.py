from collections import OrderedDict

import torch
from torch import nn

from models.bases import BaseModule
import utils
from utils.model_utils import (build_activation,
                               count_conv_flop,
                               ShuffleLayer)


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name_to_layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ZeroLayer.__name__: ZeroLayer
    }

    layer_name = layer_config.pop('name')
    layer = name_to_layer[layer_name]
    return layer.build_from_config(layer_config)


class Base2DLayer(BaseModule):
    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def _ops_list(self):
        return self._ops_order.split('_')

    @property
    def _bn_before_weight(self):
        for op in self._ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: {}'.format(self._ops_order))

    @property
    def weight_op(self):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self._in_channels,
            'out_channels': self._out_channels,
            'use_bn': self._use_bn,
            'act_func': self._act_func,
            'dropout_rate': self._dropout_rate,
            'ops_order': self._ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    @property
    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_bn: bool = True,
                 act_func: str = 'relu',
                 dropout_rate: float = 0.,
                 ops_order='weight_bn_act'):
        super(BaseModule, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._use_bn = use_bn
        self._act_func = act_func
        self._dropout_rate = dropout_rate
        self._ops_order = ops_order

        """ Modules """
        modules = {}

        # Batch Norm
        if self._use_bn:
            if self._bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None

        # Activation
        modules['act'] = build_activation(act_func=self._act_func,
                                          inplace=self._ops_list[0] != 'act')

        # Dropout
        if self._dropout_rate > 0.:
            modules['dropout'] = nn.Dropout2d(self._dropout_rate, inplace=True)
        else:
            modules['dropout'] = None

        # Weight
        modules['weight'] = self.weight_op()

        # Add modules
        for op in self._ops_list:
            if modules[op] is None:
                continue

            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])

            else:
                self.add_module(op, modules[op])

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class ConvLayer(Base2DLayer):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: [int, tuple] = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 has_shuffle: bool = False,
                 use_bn: bool = True,
                 act_func: str = 'relu',
                 dropout_rate: int = 0,
                 ops_order: str = 'weight_bn_act'):
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation
        self._groups = groups
        self._bias = bias
        self._has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        use_bn=use_bn,
                                        act_func=act_func,
                                        dropout_rate=dropout_rate,
                                        ops_order=ops_order)

    def weight_op(self):
        padding = utils.get_same_padding(self._kernel_size)
        if isinstance(padding, int):
            padding *= self._dilation
        else:
            padding[0] *= self._dilation
            padding[1] *= self._dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=padding,
            dilation=self._dilation,
            groups=self._groups,
            bias=self._bias
        )

        if self._has_shuffle and self._groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self._groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self._kernel_size, int):
            kernel_size = (self._kernel_size, self._kernel_size)
        else:
            kernel_size = self._kernel_size
        if self._groups == 1:
            if self._dilation > 1:
                return '{}x{}_DilatedConv'.format(kernel_size[0], kernel_size[1])
            else:
                return '{}x{}_Conv'.format(kernel_size[0], kernel_size[1])
        else:
            if self._dilation > 1:
                return '{}x{}_DilatedGroupConv'.format(kernel_size[0], kernel_size[1])
            else:
                return '{}x{}_GroupConv'.format(kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self._kernel_size,
            'stride': self._stride,
            'dilation': self._dilation,
            'groups': self._groups,
            'bias': self._bias,
            'has_shuffle': self._has_shuffle,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)


class DepthConvLayer(Base2DLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: [int, tuple] = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 has_shuffle: bool = False,
                 use_bn: bool = True,
                 act_func: str = 'relu',
                 dropout_rate: float = 0.,
                 ops_order: str = 'weight_bn_act'):
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilation = dilation
        self._groups = groups
        self._bias = bias
        self._has_shuffle = has_shuffle
        super(DepthConvLayer, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             use_bn=use_bn,
                                             act_func=act_func,
                                             dropout_rate=dropout_rate,
                                             ops_order=ops_order)

    def weight_op(self):
        padding = utils.get_same_padding(self._kernel_size)
        if isinstance(padding, int):
            padding *= self._dilation
        else:
            padding[0] *= self._dilation
            padding[1] *= self._dilation

        weight_dict = OrderedDict()

        weight_dict['depth_conv'] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=padding,
            dilation=self._dilation,
            groups=self._groups,
            bias=self._bias
        )

        weight_dict['point_conv'] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            groups=self._groups,
            bias=self._bias
        )

        if self._has_shuffle and self._groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self._groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self._kernel_size, int):
            kernel_size = (self._kernel_size, self._kernel_size)
        else:
            kernel_size = self._kernel_size

        if self._dilation > 1:
            return '{}x{}_DilatedGroupConv'.format(kernel_size[0], kernel_size[1])
        else:
            return '{}x{}_DepthConv'.format(kernel_size[0], kernel_size[1])

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self._kernel_size,
            'stride': self._stride,
            'dilation': self._dilation,
            'groups': self._groups,
            'bias': self._bias,
            'has_shuffle': self._has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)

    def get_flops(self, x):
        depth_flop = count_conv_flop(self.depth_conv, x)
        x = self.depth_conv(x)
        point_flop = count_conv_flop(self.point_conv, x)
        x = self.point_conv(x)
        return depth_flop + point_flop, self.forward(x)


class PoolingLayer(Base2DLayer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool_type: str,
                 kernel_size: int = 2,
                 stride: int = 2,
                 use_bn: bool = False,
                 act_func: str = None,
                 dropout_rate: float = 0.,
                 ops_order: str = 'weight_bn_act'):
        self._pool_type = pool_type
        self._kernel_size = kernel_size
        self._stride = stride
        super(PoolingLayer, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bn=use_bn,
            act_func=act_func,
            dropout_rate=dropout_rate,
            ops_order=ops_order
        )

    def weight_op(self):
        if self._stride == 1:
            # Same padding if 'stride == 1'
            padding = utils.get_same_padding(self._kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self._pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(kernel_size=self._kernel_size,
                                               stride=self._stride,
                                               padding=padding,
                                               count_include_pad=False)
        elif self._pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(kernel_size=self._kernel_size,
                                               stride=self._stride,
                                               padding=padding)
        else:
            raise NotImplementedError

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self._kernel_size, int):
            kernel_size = (self._kernel_size, self._kernel_size)
        else:
            kernel_size = self._kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self._pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self._pool_type,
            'kernel_size': self._kernel_size,
            'stride': self._stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class IdentityLayer(Base2DLayer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_bn: bool = False,
                 act_func: str = None,
                 dropout_rate: float = 0,
                 ops_order: str = 'weight_bn_act'):
        super(IdentityLayer, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bn=use_bn,
            act_func=act_func,
            dropout_rate=dropout_rate,
            ops_order=ops_order
        )

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class LinearLayer(BaseModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 use_bn: bool = False,
                 act_func: str = None,
                 dropout_rate: float = 0,
                 ops_order: str = 'weight_bn_act'):
        super(LinearLayer, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._bias = bias

        self._use_bn = use_bn
        self._act_func = act_func
        self._dropout_rate = dropout_rate
        self._ops_order = ops_order

        modules = {}

        # Batch norm
        if self._use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None

        # Activation
        modules['act'] = build_activation(self._act_func, self._ops_list[0] != 'act')

        # Dropout
        if self._dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self._dropout_rate, inplace=True)
        else:
            modules['dropout'] = None

        # Linear
        modules['weight'] = {'linear': nn.Linear(self._in_features, self._out_features, self._bias)}

        # Add modules
        for op in self._ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def _ops_list(self):
        return self._ops_order.split('_')

    @property
    def _bn_before_weight(self):
        for op in self._ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError("Invalid ops_order: {}".format(self._ops_order))

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return '{}x{}_Linear'.format(self._in_features, self._out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self._in_features,
            'out_features': self._out_features,
            'bias': self._bias,
            'use_bn': self._use_bn,
            'act_func': self._act_func,
            'dropout_rate': self._dropout_rate,
            'ops_order': self._ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

    def get_flops(self, x):
        return self.linear.weight.numel(), self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 expand_ratio: int = 6,
                 mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._kernel_size = kernel_size
        self._stride = stride
        self._expand_ratio = expand_ratio
        self._mid_channels = mid_channels

        if self._mid_channels is None:
            feature_dim = round(self._in_channels * self._expand_ratio)
        else:
            feature_dim = self._mid_channels

        if self._expand_ratio == 1:
            self._inverted_bottleneck = None
        else:
            self._inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels=self._in_channels,
                                   out_channels=feature_dim,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False)),
                ('bn', nn.BatchNorm2d(num_features=feature_dim)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = utils.get_same_padding(self._kernel_size)
        self._depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=feature_dim,
                               out_channels=feature_dim,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=pad,
                               groups=feature_dim,
                               bias=False)),
            ('bn', nn.BatchNorm2d(num_features=feature_dim)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self._point_layer = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=feature_dim,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)),
            ('bn', nn.BatchNorm2d(num_features=out_channels)),
        ]))

    def forward(self, x):
        if self._inverted_bottleneck:
            x = self._inverted_bottleneck(x)
        x = self._depth_conv(x)
        x = self._point_layer(x)
        return x

    @property
    def module_str(self):
        return '{}x{}_MBConv{}'.format(self._kernel_size,
                                       self._kernel_size,
                                       self._expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self._in_channels,
            'out_channels': self._out_channels,
            'kernel_size': self._kernel_size,
            'stride': self._stride,
            'expand_ratio': self._expand_ratio,
            'mid_channels': self._mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def get_flops(self, x):
        if self._inverted_bottleneck:
            flop_1 = count_conv_flop(self._inverted_bottleneck.conv, x)
            x = self._inverted_bottleneck(x)
        else:
            flop_1 = 0

        flop_2 = count_conv_flop(self._depth_conv.conv, x)
        x = self._depth_conv(x)

        flop_3 = count_conv_flop(self._point_linear.conv, x)
        x = self._point_linear(x)

        return flop_1 + flop_2 + flop_3, x

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(BaseModule):
    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self._stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self._stride
        w //= self._stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')

        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self._stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True
