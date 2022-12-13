from collections import defaultdict
from math import floor

import torch
from torch import nn as nn


class Bias(nn.Module):
    def __init__(self, features, spatial_dims=0):
        super().__init__()
        self.features = features
        self.spatial_dims = spatial_dims
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        return x + self.bias.view((self.features,) + (1,) * self.spatial_dims)


class Counts(defaultdict):
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(int, **kwargs)

    def __add__(self, other):
        result = self.copy()
        if isinstance(other, Counts):
            for key, value in other.items():
                result[key] += value
        else:
            for key, value in result.items():
                result[key] += other
        return result

    def __mul__(self, other):
        result = self.copy()
        for key in result:
            result[key] *= other
        return result

    def __neg__(self):
        result = self.copy()
        for key, value in result.items():
            result[key] = -value
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)


def as_tuple(x, size):
    return tuple(x) if (type(x) in [list, tuple]) else (x,) * size


def as_tuples(*args, size):
    return tuple(as_tuple(arg, size=size) for arg in args)


def conv1d_output_size(i, k, s, p, d):
    # https://arxiv.org/pdf/1603.07285.pdf
    return floor((i + 2 * p - k - (k - 1) * (d - 1)) / s) + 1


def conv2d_output_size(i, k, s, p, d):
    return tuple(conv1d_output_size(i[j], k[j], s[j], p[j], d[j]) for j in range(2))
