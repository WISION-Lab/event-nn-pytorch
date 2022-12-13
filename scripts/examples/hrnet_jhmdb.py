#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch

from datasets.jhmdb import (
    CropPerson,
    JHMDBSubset,
    InvertCropPerson,
    JOINT_PAIRS,
    N_JOINTS,
)
from eventnn.policies import ConstantThreshold
from models.hrnet import HRNetPostprocess, HRNetW32
from utils.image import write_video
from utils.misc import get_pytorch_device
from utils.visualization import draw_pose

device = get_pytorch_device()

data = JHMDBSubset(Path("data", "jhmdb"), partition=1, split="test")

model = HRNetW32(N_JOINTS).to(device)
model.load_state_dict(torch.load(Path("weights", "hrnet_jhmdb_mpii_init.pth")))
for gate in model.gates():
    gate.policy = ConstantThreshold(0.1)
model.eval()
model.counting()


def evaluate(filename=None):
    model.reset()
    model.clear_counts()
    visualization = []
    for frame, true in zip(*data[0]):
        frame = frame.unsqueeze(dim=0).to(device)
        true = true.unsqueeze(dim=0).to(device)
        cropped, _ = CropPerson()((frame, true))
        with torch.inference_mode():
            output = model(cropped)
        pred, scores = HRNetPostprocess()(output)
        pred = InvertCropPerson()(true, pred)
        if filename is not None:
            visualization.append(
                draw_pose(frame[0], pred[0], scores[0], joint_pairs=JOINT_PAIRS)
            )
    if filename is not None:
        write_video(filename, np.stack(visualization))
    return model.total_counts()["conv2d_mac"] / len(data[0][0])


model.conventional()
model.non_sparse()
ops = evaluate(
    filename=Path("outputs", "examples", "hrnet_jhmdb", "1_conventional.mp4")
)
print(f"Conventional ops: {ops:.4g}")

model.conventional()
model.sparse()
ops = evaluate()
print(f"Sparse ops: {ops:.4g}")

model.eventful()
model.sparse()
ops = evaluate(filename=Path("outputs", "examples", "hrnet_jhmdb", "1_event.mp4"))
print(f"Event ops: {ops:.4g}")
