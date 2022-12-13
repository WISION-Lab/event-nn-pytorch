import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalMNIST(Dataset):
    def __init__(
        self, filename, train=True, image_transform=None, label_transform=None
    ):
        data = np.load(filename)
        images = data["x_train"] if train else data["x_test"]
        labels = data["y_train"] if train else data["y_test"]
        self.images = torch.from_numpy(images).unsqueeze(dim=0)
        self.labels = torch.from_numpy(labels).unsqueeze(dim=0)
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        # Retrieve images (a time series).
        images = self.images[index]
        if self.image_transform is not None:
            images = self.image_transform(images)

        # Retrieve the corresponding labels (also a time series).
        labels = self.labels[index]
        if self.label_transform is not None:
            labels = self.label_transform(labels)

        return images, labels

    def __len__(self):
        return self.images.shape[0]
