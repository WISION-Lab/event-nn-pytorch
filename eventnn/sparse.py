from pathlib import Path
from sys import stderr

import torch
from torch.nn import functional as func
from torch.utils.cpp_extension import load

from eventnn.utils import as_tuples, conv2d_output_size

o_level = "-Ofast"
try:
    custom_conv2d_cpp = load(
        name="custom_conv2d_cpp",
        sources=[str(Path("cpp", "custom_conv2d.cpp"))],
        extra_cflags=[o_level],
    )
    custom_sparse_conv2d_cpp = load(
        name="custom_sparse_conv2d_cpp",
        sources=[str(Path("cpp", "custom_sparse_conv2d.cpp"))],
        extra_cflags=[o_level],
    )
except RuntimeError as e:
    print(
        f'Warning: Could not load custom C++ operators.\nError message: "{e}"',
        file=stderr,
    )


def custom_conv2d(x, weight, stride=1, padding=0, dilation=1):
    stride, padding, dilation = as_tuples(stride, padding, dilation, size=2)
    if dilation != (1, 1):
        raise NotImplementedError("Dilation not supported in custom_conv2d.")
    x = func.pad(x, padding[1:] * 2 + padding[:1] * 2)
    return custom_conv2d_cpp.forward(x, weight, stride)


def custom_sparse_conv2d(x, weight, stride=1, padding=0, dilation=1):
    stride, padding, dilation = as_tuples(stride, padding, dilation, size=2)
    if dilation != (1, 1):
        raise NotImplementedError("Dilation not supported in custom_sparse_conv2d.")
    x = func.pad(x, padding[1:] * 2 + padding[:1] * 2)
    return custom_sparse_conv2d_cpp.forward(x, weight, stride)


def unrolled_conv2d(x, weight, stride=1, padding=0, dilation=1):
    return _unrolled_conv2d(
        x, weight, stride=stride, padding=padding, dilation=dilation, sparse=False
    )


def unrolled_sparse_conv2d(x, weight, stride=1, padding=0, dilation=1):
    return _unrolled_conv2d(
        x, weight, stride=stride, padding=padding, dilation=dilation, sparse=True
    )


def _unrolled_conv2d(x, weight, stride=1, padding=0, dilation=1, sparse=False):
    stride, padding, dilation = as_tuples(stride, padding, dilation, size=2)
    size = conv2d_output_size(x.shape[2:], weight.shape[2:], stride, padding, dilation)
    x = func.unfold(
        x, weight.shape[2:], stride=stride, padding=padding, dilation=dilation
    )
    weight = weight.view(weight.shape[0], -1)

    if sparse:
        output = torch.zeros(
            (x.shape[0], weight.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device
        )
        for x_i, output_i in zip(x, output):
            nonzero = x_i.any(dim=0)
            output_i[:, nonzero] = weight.matmul(x_i[:, nonzero])
    else:
        output = weight.matmul(x)

    return output.view(output.shape[:2] + size)
