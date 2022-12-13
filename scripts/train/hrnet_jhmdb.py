#!/usr/bin/env python3

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from sys import stderr

import torch
import torch.optim as optim
import yaml
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from datasets.jhmdb import (
    N_JOINTS,
    CreateHeatmap,
    CropPerson,
    JHMDBSubset,
    TrainingAugment,
)
from eventnn.counted import CountedConv2d
from models.hrnet import HRNetPostprocess, HRNetW32
from utils.metrics import MeanValue, PCK
from utils.misc import get_pytorch_device


def main(args):
    device = get_pytorch_device()
    torch.random.manual_seed(42)

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    augmentation = TrainingAugment(
        scale_min=1.0 - config["augmentation"]["scale_range"],
        scale_max=1.0 + config["augmentation"]["scale_range"],
        theta_min=-config["augmentation"]["theta_range"],
        theta_max=+config["augmentation"]["theta_range"],
    )
    train_transforms = Compose(
        [augmentation, CropPerson(), CreateHeatmap(sigma=config["heatmap_sigma"])]
    )
    train = JHMDBSubset(
        Path("data", "jhmdb"),
        partition=args.partition,
        split="train",
        flatten=True,
        val_fraction=config["val_fraction"],
        combined_transform=train_transforms,
    )
    train = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    val_transforms = Compose(
        [CropPerson(), CreateHeatmap(sigma=config["heatmap_sigma"])]
    )
    val = JHMDBSubset(
        Path("data", "jhmdb"),
        partition=args.partition,
        split="val",
        flatten=True,
        val_fraction=config["val_fraction"],
        combined_transform=val_transforms,
    )
    val = DataLoader(val, batch_size=config["batch_size"], shuffle=False)
    postprocess = HRNetPostprocess()

    model = HRNetW32(N_JOINTS).to(device)
    if "initial_weights_path" in config:
        model.load_state_dict(torch.load(Path(*config["initial_weights_path"])))
    else:
        model.initialize_xavier_uniform()

    loss_function = MSELoss()
    weight_decay = config["optimizer"].get("weight_decay", None)
    regular_params = set(model.parameters())
    decayed_params = set()
    optimizer_base_args = config["optimizer"].get("arguments", {})
    if weight_decay is not None:
        optimizer_base_args["weight_decay"] = 0.0
        for conv_layer in model.modules_of_type(CountedConv2d):
            regular_params.remove(conv_layer.weight)
            decayed_params.add(conv_layer.weight)
    optimizer = getattr(optim, config["optimizer"]["type"])(
        list(regular_params), config["learning_rates"]["start"], **optimizer_base_args
    )
    if weight_decay is not None:
        optimizer.add_param_group(
            {"params": list(decayed_params), "weight_decay": weight_decay}
        )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        config["learning_rates"]["steps"],
        gamma=config["learning_rates"]["decay"],
    )
    mean_loss = MeanValue()
    pck = PCK()

    now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard = SummaryWriter(
        str(Path("tensorboard", f"hrnet_jhmdb_{args.name}_{now_str}"))
    )

    best_pck = 0.0
    for epoch in range(config["epochs"]):
        print(f'Epoch {epoch + 1}/{config["epochs"]}', flush=True)

        model.train()
        pck.reset()
        mean_loss.reset()
        for i, (inputs, true, heatmaps) in enumerate(train):
            print(f"Batch {i + 1}/{len(train)}\r", end="", file=stderr, flush=True)
            outputs = model(inputs.to(device))
            loss = loss_function(heatmaps.to(device), outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss.update(loss.item())
            pred, _ = postprocess(outputs)
            pck.update(true.to(device), pred, inputs.shape[-2:])
        tensorboard.add_scalars("loss", {"train": mean_loss.compute()}, epoch)
        tensorboard.add_scalars("PCK", {"train": pck.compute()}, epoch)
        print(f"Training loss: {mean_loss.compute():.4g}", flush=True)
        print(f"Training PCK: {pck.compute():.4f}", flush=True)

        if len(val) > 0:
            model.eval()
            pck.reset()
            mean_loss.reset()
            for i, (inputs, true, heatmaps) in enumerate(val):
                print(f"Batch {i + 1}/{len(val)}\r", end="", file=stderr, flush=True)
                outputs = model(inputs.to(device))
                loss = loss_function(heatmaps.to(device), outputs)
                mean_loss.update(loss.item())
                pred, _ = postprocess(outputs)
                pck.update(true.to(device), pred, inputs.shape[-2:])
            tensorboard.add_scalars("loss", {"val": mean_loss.compute()}, epoch)
            tensorboard.add_scalars("PCK", {"val": pck.compute()}, epoch)
            print(f"Validation loss: {mean_loss.compute():.4g}", flush=True)
            print(f"Validation PCK: {pck.compute():.4f}", flush=True)
            if pck.compute() > best_pck:
                best_pck = pck.compute()
                path = Path(args.weights)
                path = path.with_stem(f"{path.stem}_best")
                torch.save(model.state_dict(), path)
                print(f"Saved weights to {path}.", flush=True)

        print("", flush=True)
        scheduler.step()

    tensorboard.close()
    path = Path(args.weights)
    path = path.with_stem(f"{path.stem}_last")
    torch.save(model.state_dict(), path)
    print(f"Saved weights to {path}.", flush=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "partition", type=int, choices=[1, 2, 3], help="the train/test partition to use"
    )
    parser.add_argument("config", help="the .yml configuration file")
    parser.add_argument("name", help="a name for the Tensorboard directory")
    parser.add_argument("weights", help="the location to save output weights")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
