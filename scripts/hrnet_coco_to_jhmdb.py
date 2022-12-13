#!/usr/bin/env python3

from pathlib import Path

import torch

from datasets.jhmdb import COCO_REMAP_INDICES

weights = torch.load(Path("weights", "hrnet_coco.pth"))
for key in "final.0.weight", "final.2.bias":
    weights[key] = weights[key][COCO_REMAP_INDICES]
torch.save(weights, Path("weights", "hrnet_jhmdb_coco_init.pth"))
