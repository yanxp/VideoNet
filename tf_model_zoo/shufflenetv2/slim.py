#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def g_name(g_name, m):
    m.g_name = g_name
    return m

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = x.reshape(x.shape[0], self.groups, x.shape[1] // self.groups, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.ShuffleChannel(layer, group=self.groups)
        caffe_net[self.g_name] = layer
        return layer


def channel_shuffle(name, groups):
    return g_name(name, ChannelShuffle(groups))


class Permute(nn.Module):
    def __init__(self, order):
        super(Permute, self).__init__()
        self.order = order

    def forward(self, x):
        x = x.permute(*self.order).contiguous()
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.Permute(layer, order=list(self.order))
        caffe_net[self.g_name] = layer
        return layer


def permute(name, order):
    return g_name(name, Permute(order))


class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        assert self.axis == 1
        x = x.reshape(x.shape[0], -1)
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = L.Flatten(layer, axis=self.axis)
        caffe_net[self.g_name] = layer
        return layer


def flatten(name, axis):
    return g_name(name, Flatten(axis))

def conv_bn_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
        g_name(name + '/relu', nn.ReLU(inplace=True)),
    )


def conv_bn(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        g_name(name + '/bn', nn.BatchNorm2d(out_channels)),
    )


def conv(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True))


def conv_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)),
        g_name(name + '/relu', nn.ReLU()),
    )

def conv_prelu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        g_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)),
        g_name(name + '/prelu', nn.PReLU()),
    )
    
