import shutil
from pathlib import Path
from random import Random

import torch
import torch.nn as nn
import torchvision.transforms.functional as transforms
from scipy.io import loadmat
from torch.utils.data import Dataset

from datasets import coco, mpii
from utils.image import as_float32, centered_pad_and_crop, read_image
from utils.misc import rotate_points, torch_rand_uniform

JOINT_NAMES = [
    "neck",
    "belly",
    "face",
    "right shoulder",
    "left shoulder",
    "right hip",
    "left hip",
    "right elbow",
    "left elbow",
    "right knee",
    "left knee",
    "right wrist",
    "left wrist",
    "right ankle",
    "left ankle",
]

# For visualization purposes
JOINT_PAIRS = [
    ("face", "neck"),
    ("neck", "belly"),
    ("neck", "right shoulder"),
    ("neck", "left shoulder"),
    ("right shoulder", "right elbow"),
    ("left shoulder", "left elbow"),
    ("right elbow", "right wrist"),
    ("left elbow", "left wrist"),
    ("belly", "right hip"),
    ("belly", "left hip"),
    ("right hip", "right knee"),
    ("left hip", "left knee"),
    ("right knee", "right ankle"),
    ("left knee", "left ankle"),
]
JOINT_PAIRS = [
    (JOINT_NAMES.index(s_1), JOINT_NAMES.index(s_2)) for s_1, s_2 in JOINT_PAIRS
]

# JOINT_NAMES[i] corresponds to COCO_REMAP_NAMES[i].
COCO_REMAP_NAMES = [
    "left ear",
    "left hip",
    "nose",
    "right shoulder",
    "left shoulder",
    "right hip",
    "left hip",  # Duplicate
    "right elbow",
    "left elbow",
    "right knee",
    "left knee",
    "right wrist",
    "left wrist",
    "right ankle",
    "left ankle",
]

# JHMDB joint i corresponds to COCO joint COCO_REMAP_INDICES[i].
COCO_REMAP_INDICES = [coco.JOINT_NAMES.index(name) for name in COCO_REMAP_NAMES]

# JOINT_NAMES[i] corresponds to MPII_REMAP_NAMES[i].
MPII_REMAP_NAMES = [
    "upper neck",
    "pelvis",
    "head top",
    "right shoulder",
    "left shoulder",
    "right hip",
    "left hip",
    "right elbow",
    "left elbow",
    "right knee",
    "left knee",
    "right wrist",
    "left wrist",
    "right ankle",
    "left ankle",
]

# JHMDB joint i corresponds to MPII joint MPII_REMAP_INDICES[i].
MPII_REMAP_INDICES = [mpii.JOINT_NAMES.index(name) for name in MPII_REMAP_NAMES]

N_JOINTS = 15


# See function generate_target in
# https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/lib/dataset/JointsDataset.py.
class CreateHeatmap(nn.Module):
    def __init__(self, sigma=2.0, scale=4):
        super().__init__()
        self.sigma = sigma
        self.scale = scale

    def forward(self, inputs_and_labels):
        inputs, labels = inputs_and_labels
        multiple_items = labels.dim() > 2
        if not multiple_items:
            inputs = inputs.unsqueeze(dim=0)
            labels = labels.unsqueeze(dim=0)

        heatmap_size = (
            int(inputs.shape[-2] / self.scale),
            int(inputs.shape[-1] / self.scale),
        )
        heatmaps = torch.zeros(labels.shape[:2] + heatmap_size, dtype=inputs.dtype)
        y, x = torch.meshgrid(
            torch.arange(heatmap_size[0]), torch.arange(heatmap_size[1]), indexing="ij"
        )
        for labels_i, heatmaps_i in zip(labels, heatmaps):
            for joint, heatmap in zip(labels_i, heatmaps_i):
                j_x, j_y = joint / self.scale
                heatmap += torch.exp(
                    -((x - j_x) ** 2 + (y - j_y) ** 2) / (2.0 * self.sigma ** 2)
                )

        if multiple_items:
            return inputs, labels, heatmaps
        else:
            return inputs[0], labels[0], heatmaps[0]


# See line 227 of
# https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/lib/dataset/coco.py
# for default buffer value.
class CropPerson(nn.Module):
    def __init__(self, fixed_size=(256, 256), buffer=0.25):
        super().__init__()
        self.fixed_size = fixed_size
        self.buffer = buffer

    def forward(self, inputs_and_labels):
        inputs, labels = inputs_and_labels
        labels = labels.clone()
        multiple_items = labels.dim() > 2
        if not multiple_items:
            inputs = inputs.unsqueeze(dim=0)
            labels = labels.unsqueeze(dim=0)

        positions, box_sizes = find_bounding_boxes(labels, self.buffer)
        cropped = []
        for i, (frame, position, size) in enumerate(zip(inputs, positions, box_sizes)):
            frame = transforms.crop(frame, position[1], position[0], size[1], size[0])
            cropped.append(centered_pad_and_crop(frame, self.fixed_size))
            labels[i] -= position
            labels[i, :, 0] += (self.fixed_size[1] - size[0].item()) // 2
            labels[i, :, 1] += (self.fixed_size[0] - size[1].item()) // 2
        inputs = torch.stack(cropped)

        if multiple_items:
            return inputs, labels
        else:
            return inputs[0], labels[0]


# Returns the labels to the original image coordinates.
class InvertCropPerson(nn.Module):
    def __init__(self, fixed_size=(256, 256), buffer=0.25):
        super().__init__()
        self.fixed_size = fixed_size
        self.buffer = buffer

    def forward(self, true, pred):
        pred = pred.clone()
        multiple_items = pred.dim() > 2
        if not multiple_items:
            true = true.unsqueeze(dim=0)
            pred = pred.unsqueeze(dim=0)

        positions, sizes = find_bounding_boxes(true, self.buffer)
        for i, (position, size) in enumerate(zip(positions, sizes)):
            pred[i, :, 0] -= (self.fixed_size[1] - size[0].item()) // 2
            pred[i, :, 1] -= (self.fixed_size[0] - size[1].item()) // 2
            pred[i] += position

        if multiple_items:
            return pred
        else:
            return pred[0]


class JHMDBSubset(Dataset):
    def __init__(
        self,
        location,
        partition,
        split="train",
        flatten=False,
        val_fraction=0.0,
        val_seed=42,
        input_transform=None,
        label_transform=None,
        combined_transform=None,
    ):
        if partition not in (1, 2, 3):
            raise ValueError("The partition must be 1, 2, or 3.")
        if split not in ("train", "test", "val"):
            raise ValueError('The split must be "train", "test", or "val".')

        if not self.is_unpacked(location):
            self.clean(location)
            self.unpack(location)

        self.location = location
        self.flatten = flatten
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.combined_transform = combined_transform

        train_val = split in ("train", "val")
        video_ids = self.get_video_ids(location, partition, train_val)
        if train_val:
            rng = Random()
            rng.seed(val_seed)
            rng.shuffle(video_ids)
            i = int(len(video_ids) * val_fraction)
            video_ids = video_ids[:i] if (split == "val") else video_ids[i:]
        video_ids.sort()
        self.video_ids = video_ids

        self.video_lengths = [
            len(list(Path(self.location, "Rename_Images", video_id).glob("*.png")))
            for video_id in self.video_ids
        ]
        if flatten:
            self.n_items = sum(self.video_lengths)
            self.lookup = []
            for i, length in enumerate(self.video_lengths):
                for j in range(length):
                    self.lookup.append((i, j))
        else:
            self.n_items = len(self.video_lengths)

    def __getitem__(self, index):
        # Read and preprocess the input (image/video).
        path = Path(self.location, "Rename_Images")
        f = "{:05d}.png"
        if self.flatten:
            i, j = self.lookup[index]
            video_id = self.video_ids[i]
            inputs = read_image(path / video_id / f.format(j + 1))
        else:
            video_id = self.video_ids[index]
            n = self.video_lengths[index]
            inputs = torch.stack(
                [read_image(path / video_id / f.format(j + 1)) for j in range(n)]
            )
        inputs = as_float32(inputs)
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        # Read and preprocess the labels (joint positions).
        path = Path(self.location, "joint_positions", video_id, "joint_positions.mat")
        labels = loadmat(str(path))["pos_img"]
        labels = torch.from_numpy(labels).float().permute(2, 1, 0)
        if self.flatten:
            # noinspection PyUnboundLocalVariable
            labels = labels[j]
        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.combined_transform is not None:
            return self.combined_transform((inputs, labels))
        else:
            return inputs, labels

    def __len__(self):
        return self.n_items

    @staticmethod
    def clean(location):
        # Remove the indicator file.
        Path(location, "unpacked").unlink(missing_ok=True)

        # Remove any directories produced by unpack.
        for destination in [
            "estimated_joint_positions",
            "joint_positions",
            "ReCompress_Videos",
            "Rename_Images",
            "splits",
            "sub_splits",
        ]:
            path = Path(location, destination)
            if path.is_dir():
                shutil.rmtree(path)

    @staticmethod
    def get_video_ids(location, split, train):
        video_ids = []
        for filename in sorted(
            Path(location, "sub_splits").glob(f"*_test_split_{split}.txt")
        ):
            # The name of the text file corresponds to the names of the
            # folders containing videos and annotations.
            base = Path(filename).stem.removesuffix(f"_test_split_{split}")

            # Each line in the text file contains the video name and the
            # train/test assignment (1 for train, 2 for test).
            with open(filename, "r") as file:
                for line in file:
                    tokens = line.strip().split(" ")
                    if train == (tokens[-1] == "1"):
                        # Use .stem to remove the .avi file extension.
                        video_ids.append(str(Path(base, Path(tokens[0]).stem)))
        return video_ids

    @staticmethod
    def is_unpacked(location):
        return Path(location, "unpacked").is_file()

    @staticmethod
    def unpack(location):
        # Unpack .zip and .tar.gz archives.
        for archive, destination in [
            ("estimated_joint_positions.zip", location),
            ("JHMDB_video.zip", location),
            ("joint_positions.zip", location),
            ("Rename_Images.tar.gz", location),
            ("splits.zip", location),
            ("sub_splits.zip", Path(location, "sub_splits")),
        ]:
            print(f"Unpacking {archive}...", flush=True)
            shutil.unpack_archive(Path(location, archive), destination)
        shutil.rmtree(Path(location, "__MACOSX"))
        print("Done.", flush=True)

        # Create an empty indicator file.
        Path(location, "unpacked").touch()


class TrainingAugment(nn.Module):
    def __init__(self, scale_min=0.6, scale_max=1.4, theta_min=-40.0, theta_max=40.0):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.theta_min = theta_min
        self.theta_max = theta_max

    def forward(self, inputs_and_labels):
        inputs, labels = inputs_and_labels
        labels = labels.clone()

        # Random scaling
        scale = torch_rand_uniform(self.scale_min, self.scale_max)
        inputs = transforms.resize(
            inputs,
            [int(scale * x) for x in inputs.shape[-2:]],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        labels *= scale

        # Random rotation
        theta = torch_rand_uniform(self.theta_min, self.theta_max)
        labels = rotate_points(
            labels,
            theta,
            center_x=inputs.shape[-1] / 2.0,
            center_y=inputs.shape[-2] / 2.0,
        )
        inputs = transforms.rotate(
            inputs,
            float(theta.item()),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        # Random horizontal flipping
        if torch.rand(()) > 0.5:
            inputs = transforms.hflip(inputs)
            labels[..., 0] = -labels[..., 0] + inputs.shape[-1]

        return inputs, labels


def find_bounding_boxes(labels, buffer):
    min_ = labels.amin(dim=-2)
    max_ = labels.amax(dim=-2)
    sizes = (max_ - min_) * (1.0 + buffer)
    positions = (max_ + min_) / 2.0 - sizes / 2.0
    return positions.int(), sizes.int()
