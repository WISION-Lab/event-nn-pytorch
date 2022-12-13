#!/usr/bin/env python3

from pathlib import Path

import torch

from datasets.temporal_mnist import TemporalMNIST
from eventnn.policies import ConstantThreshold
from models.dense import Dense
from utils.misc import get_pytorch_device

device = get_pytorch_device()

data = TemporalMNIST(Path("data", "temporal_mnist.npz"), train=False)

model = Dense().to(device)
model.load_state_dict(torch.load(Path("weights", "dense.pth")))
for gate in model.gates():
    gate.policy = ConstantThreshold(0.2)
model.eval()
model.counting()


def evaluate():
    model.reset()
    model.clear_counts()
    images, labels = data[0]
    images = images.to(device)
    labels = labels.to(device)
    correct = 0
    with torch.inference_mode():
        for image, label in zip(images, labels):
            output = model(image)
            correct += int(output.argmax(dim=-1) == label)
    print(f"Accuracy: {100 * correct / len(images):.2f}%", flush=True)
    print(f"Ops: {model.total_counts()['linear_mac'] / len(images):.4g}", flush=True)


print("Conventional", flush=True)
model.conventional()
model.non_sparse()
evaluate()

print("\nSparse", flush=True)
model.conventional()
model.sparse()
evaluate()

print("\nEvent", flush=True)
model.eventful()
model.sparse()
evaluate()
