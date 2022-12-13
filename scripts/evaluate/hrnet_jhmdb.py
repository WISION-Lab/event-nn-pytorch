#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from sys import stderr

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.jhmdb import N_JOINTS, CropPerson, JHMDBSubset
from eventnn.policies import ConstantThreshold
from models.hrnet import HRNetPostprocess, HRNetW32, HRNetW32Flip
from utils.metrics import PCK
from utils.misc import get_pytorch_device


def main(args):
    device = get_pytorch_device()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    test = JHMDBSubset(
        Path("data", "jhmdb"),
        partition=args.partition,
        split="test",
        flatten=False,
        combined_transform=CropPerson(),
    )
    test = DataLoader(test, batch_size=1)
    postprocess = HRNetPostprocess()

    model = (HRNetW32Flip if config["flip"] else HRNetW32)(N_JOINTS).to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.counting()

    def evaluate():
        n_frames = 0
        metric = PCK()
        model.clear_counts()
        with torch.inference_mode():
            for i, (inputs, true) in enumerate(test):
                print(f"Item {i + 1}/{len(test)}\r", end="", file=stderr, flush=True)
                model.reset()
                for t in range(inputs.shape[1]):
                    n_frames += 1
                    outputs = model(inputs[:, t].to(device))
                    pred, _ = postprocess(outputs)
                    metric.update(true[:, t].to(device), pred, inputs.shape[-2:])
        print(f"PCK: {metric.compute():.4f}", flush=True)
        print(f'Ops: {model.total_counts()["conv2d_mac"] / n_frames:.4g}', flush=True)

    print("Conventional", flush=True)
    model.conventional()
    model.non_sparse()
    evaluate()

    print("\nSparse", flush=True)
    model.conventional()
    model.sparse()
    evaluate()

    for threshold in config["thresholds"]["event"]:
        print(f"\nEvent with threshold {threshold}", flush=True)
        for gate in model.gates():
            gate.policy = ConstantThreshold(threshold)
        model.eventful()
        model.long_term_memory()
        model.sparse()
        evaluate()

    for threshold in config["thresholds"]["memory_loss"]:
        print(f"\nMemory loss with threshold {threshold}", flush=True)
        for gate in model.gates():
            gate.policy = ConstantThreshold(threshold)
        model.eventful()
        model.memory_loss()
        model.sparse()
        evaluate()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "partition", type=int, choices=[1, 2, 3], help="the train/test partition to use"
    )
    parser.add_argument("config", help="the .yml configuration file")
    parser.add_argument("weights", help="the location of the model weights")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
