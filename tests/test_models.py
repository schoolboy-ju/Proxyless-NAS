import pytest

import torch

from models.layers import *


@pytest.fixture()
def sample_data():
    return torch.rand((1, 3, 32, 32))


@pytest.fixture()
def sample_linear_data():
    return torch.rand((1, 128))


def test_conv_layer(sample_data):
    model = ConvLayer(in_channels=3, out_channels=16)
    out = model(sample_data)
    print(out.shape)


def test_depth_conv_layer(sample_data):
    model = DepthConvLayer(in_channels=3, out_channels=16)
    out = model(sample_data)
    print(out.shape)


def test_pooling_layer(sample_data):
    model = PoolingLayer(in_channels=3, out_channels=16, pool_type='avg')
    out = model(sample_data)
    print(out.shape)


def test_identity_layer(sample_data):
    model = IdentityLayer(in_channels=3, out_channels=16)
    out = model(sample_data)
    print(out.shape)


def test_linear_layer(sample_linear_data):
    model = LinearLayer(in_features=128, out_features=10)
    out = model(sample_linear_data)
    print(out.shape)


def test_mb_inverted_conv_layer(sample_data):
    model = MBInvertedConvLayer(in_channels=3, out_channels=16)
    out = model(sample_data)
    print(out.shape)


def test_identity_layer(sample_data):
    model = ZeroLayer(stride=1)
    out = model(sample_data)
    print(out.shape)
