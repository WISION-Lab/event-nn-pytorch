import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torchvision.transforms.functional import hflip

from eventnn.core import Accumulator, EventModule, Gate
from eventnn.counted import CountedConv2d
from eventnn.utils import Bias
from utils.image import pad_to_multiple

# Based on the original authors' official implementation:
# https://github.com/HRNet/HRNet-Human-Pose-Estimation
# Download weights from Google Drive:
# https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing

BN_MOMENTUM = 0.1
EXPAND_RATIO = 4
N_BLOCKS = 4


class BasicBlock(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv_1 = ConvBlock(c_in, c_in)
        self.conv_2 = ConvBlock(c_in, c_in, relu=False, gate=False)
        self.accumulator = Accumulator()
        self.relu = nn.ReLU()
        self.gate = Gate()

    def forward(self, x):
        skip = self.accumulator(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x + skip
        x = self.relu(x)
        x = self.gate(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        c_expand = EXPAND_RATIO * c_out
        if c_in != c_expand:
            self.skip_layer = ConvBlock(
                c_in, c_expand, kernel_size=1, relu=False, gate=False
            )
        else:
            self.skip_layer = Accumulator()
        self.conv_1 = ConvBlock(c_in, c_out, kernel_size=1)
        self.conv_2 = ConvBlock(c_out, c_out)
        self.conv_3 = ConvBlock(c_out, c_expand, kernel_size=1, relu=False, gate=False)
        self.relu = nn.ReLU()
        self.gate = Gate()

    def forward(self, x):
        skip = self.skip_layer(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x + skip
        x = self.relu(x)
        x = self.gate(x)
        return x


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        stride=1,
        bias=False,
        bn=True,
        relu=True,
        gate=True,
    ):
        super().__init__(
            CountedConv2d(
                c_in, c_out, kernel_size, stride=stride, padding=(kernel_size - 1) // 2
            ),
            Accumulator(),
        )
        if bias:
            self.append(Bias(c_out, spatial_dims=2))
        if bn:
            self.append(nn.BatchNorm2d(c_out, momentum=BN_MOMENTUM))
        if relu:
            self.append(nn.ReLU())
        if gate:
            self.append(Gate())


class FuseBlock(nn.Module):
    def __init__(self, c_list, multi_scale):
        super().__init__()
        self.layers = nn.ModuleList()
        self.final = nn.ModuleList()
        n_outputs = len(c_list) if multi_scale else 1
        for i, c_out in enumerate(c_list[:n_outputs]):
            self.layers.append(nn.ModuleList())
            self.final.append(nn.Sequential(Accumulator(), nn.ReLU(), Gate()))
            layer_i = self.layers[i]

            # Downsample using strided convolution.
            for j, c_in in enumerate(c_list[:i]):
                layer_i.append(nn.Sequential())
                for k in range(i - j - 1):
                    layer_i[j].append(ConvBlock(c_in, c_in, stride=2))
                layer_i[j].append(ConvBlock(c_in, c_out, stride=2, relu=False))

            # Do nothing.
            layer_i.append(nn.Identity())

            # Upsample using nearest neighbor interpolation.
            for j, c_in in enumerate(c_list[i + 1 :], start=i + 1):
                layer_i.append(
                    nn.Sequential(
                        ConvBlock(c_in, c_out, kernel_size=1, relu=False),
                        nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                    )
                )

    def forward(self, x):
        fused = []
        for layer_i, final_i in zip(self.layers, self.final):
            # noinspection PyTypeChecker
            y = sum(f(x_j) for f, x_j in zip(layer_i, x))
            fused.append(final_i(y))
        return fused


class HRModule(nn.Module):
    def __init__(self, c_list, multi_scale=True):
        super().__init__()
        self.branches = nn.ModuleList()
        for c in c_list:
            self.branches.append(
                nn.Sequential(*[BasicBlock(c) for _ in range(N_BLOCKS)])
            )
        self.fuse = FuseBlock(c_list, multi_scale)

    def forward(self, x):
        x = [f(x_i) for f, x_i in zip(self.branches, x)]
        x = self.fuse(x)
        return x


class HRNetPostprocess(nn.Module):
    def __init__(self, scale=4, offset=0.25):
        super().__init__()
        self.scale = scale
        self.offset = offset

    def forward(self, x):
        # Find the best and second-best points.
        scores, i = x.view(x.shape[:-2] + (-1,)).topk(k=2, dim=-1)
        coords_x = i % x.shape[-1]
        coords_y = i.div(x.shape[-1], rounding_mode="floor")
        coords = torch.stack([coords_x, coords_y], dim=-2).float()
        joints = coords[..., 0]
        scores = scores[..., 0]

        # Adjust the joint coordinates in the direction of the
        # second-best point.
        deltas = coords[..., 1] - joints
        deltas /= deltas.norm(dim=-1, keepdim=True)
        joints += self.offset * deltas

        return self.scale * joints, scores


# Assumes input values in the range [0, 1].
class HRNetW32(EventModule):
    def __init__(self, n_joints):
        super().__init__()

        # Preprocessing
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Stem
        self.input_gate = Gate()
        self.stem_1 = ConvBlock(c_in=3, c_out=64, stride=2)
        self.stem_2 = ConvBlock(c_in=64, c_out=64, stride=2)
        self.stem_3 = nn.Sequential(
            *[BottleneckBlock(c_in=c, c_out=64) for c in [64, 256, 256, 256]]
        )

        # HR stages
        c_list_1 = [256]
        c_list_2 = [32, 64]
        c_list_3 = [32, 64, 128]
        c_list_4 = [32, 64, 128, 256]
        self.hr_stage_2 = HRStage(c_list_1, c_list_2, n_modules=1)
        self.hr_stage_3 = HRStage(c_list_2, c_list_3, n_modules=4)
        self.hr_stage_4 = HRStage(c_list_3, c_list_4, n_modules=3, multi_scale=False)

        # Output
        self.final = ConvBlock(
            c_list_4[0],
            n_joints,
            kernel_size=1,
            bias=True,
            bn=False,
            gate=False,
            relu=False,
        )

    def forward(self, x):
        x = pad_to_multiple(x, 32)
        x = self.normalize(x)
        x = self.input_gate(x)
        x = self.stem_1(x)
        x = self.stem_2(x)
        x = self.stem_3(x)
        x = [x]
        x = self.hr_stage_2(x)
        x = self.hr_stage_3(x)
        x = self.hr_stage_4(x)
        x = self.final(x[0])
        return x

    def initialize_xavier_uniform(self):
        for module in self.modules_of_type(CountedConv2d):
            nn.init.xavier_uniform_(module.weight)


# Note that the two submodules branch_1 and branch_2 do *not* share
# weights. Because of this, it is not recommended to use this class
# during training. The reason we use two separate submodules is to allow
# independent states when treating the model as an EvNet.
class HRNetW32Flip(EventModule):
    def __init__(self, n_joints):
        super().__init__()
        self.branch_1 = HRNetW32(n_joints)
        self.branch_2 = HRNetW32(n_joints)

    def forward(self, x):
        # This padding function is not symmetric/centered, so we need to
        # pad first. Otherwise, the padded images within the two
        # branches may be different.
        x = pad_to_multiple(x, 32)

        # Average the outputs for the original and flipped images.
        output_1 = self.branch_1(x)
        output_2 = hflip(self.branch_2(hflip(x)))
        return (output_1 + output_2) / 2.0

    def load_state_dict(self, state_dict, strict=True):
        self.branch_1.load_state_dict(state_dict, strict=strict)
        self.branch_2.load_state_dict(state_dict, strict=strict)


class HRStage(nn.Module):
    def __init__(self, c_in_list, c_out_list, n_modules, multi_scale=True):
        super().__init__()

        # Inter-stage transition
        self.transition = nn.ModuleList()
        for c_in, c_out in zip(c_in_list, c_out_list[:-1]):
            if c_in != c_out:
                self.transition.append(ConvBlock(c_in, c_out))
            else:
                self.transition.append(nn.Identity())
        self.transition.append(ConvBlock(c_in_list[-1], c_out_list[-1], stride=2))

        # Stack of HR modules
        self.hr_modules = nn.Sequential()
        for i in range(n_modules):
            self.hr_modules.append(
                HRModule(c_out_list, multi_scale or i != n_modules - 1)
            )

    def forward(self, x):
        x = [f(x_i) for f, x_i in zip(self.transition, x + x[-1:])]
        x = self.hr_modules(x)
        return x
