from math import prod

import torch
import torch.nn as nn
from torch.nn import functional as func

from eventnn.core import EventModule
from eventnn.sparse import (
    custom_conv2d,
    custom_sparse_conv2d,
    unrolled_conv2d,
    unrolled_sparse_conv2d,
)
from eventnn.utils import as_tuples


# Does not include bias computation. Use the standalone
# eventnn.utils.Bias module if you need a bias.
class CountedConv2d(EventModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        super().__init__()
        kernel_size, stride, padding, dilation = as_tuples(
            kernel_size, stride, padding, dilation, size=2
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.zeros((out_channels, in_channels) + self.kernel_size)
        )
        self.conv_mode = "standard"

    def forward(self, x):
        if self.count_mode:
            if self.sparse_mode:
                n_updates = x.count_nonzero().item()
            else:
                n_updates = prod(x.shape)

            # Assumes that each input pixel gets hit the same number of
            # times. This is an approximation - in reality padding and
            # certain strides will cause some input pixels to be hit
            # more often then others. But this is probably good enough.
            # If we want something exact, we can evaluate
            # torch.sum(self.conv2d(x != 0)) with the conv2d kernel
            # replaced by all 1's. But, this is expensive to compute -
            # probably more expensive than the original convolution!
            fan_out = prod(self.kernel_size) * self.out_channels
            stride_ratio = prod(self.stride)
            self.counts["conv2d_mac"] += n_updates * fan_out / stride_ratio

        args = [x, self.weight]
        kwargs = dict(stride=self.stride, padding=self.padding, dilation=self.dilation)
        if self.conv_mode == "standard":
            return func.conv2d(*args, **kwargs)
        elif self.conv_mode == "unrolled":
            if self.sparse_mode:
                return unrolled_sparse_conv2d(*args, **kwargs)
            else:
                return unrolled_conv2d(*args, **kwargs)
        elif self.conv_mode == "custom":
            if self.sparse_mode:
                return custom_sparse_conv2d(*args, **kwargs)
            else:
                return custom_conv2d(*args, **kwargs)
        else:
            raise ValueError('conv_mode must be "standard", "unrolled", or "custom."')


# Does not include bias computation. Use the standalone
# eventnn.utils.Bias module if you need a bias.
class CountedLinear(EventModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        if self.count_mode:
            if self.sparse_mode:
                n_updates = x.count_nonzero().item()
            else:
                n_updates = prod(x.shape)
            self.counts["linear_mac"] += n_updates * self.linear.out_features
        return self.linear(x)
