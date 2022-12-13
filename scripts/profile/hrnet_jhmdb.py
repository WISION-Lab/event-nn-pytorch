#!/usr/bin/env python3

import time
from pathlib import Path

import torch
from torch.profiler import profile

from datasets.jhmdb import N_JOINTS, CropPerson, JHMDBSubset
from eventnn.counted import CountedConv2d
from eventnn.policies import ConstantThreshold
from models.hrnet import HRNetW32

# To ensure a fair comparison between PyTorch and custom operators
# (which are not currently multithreaded)
torch.set_num_threads(1)

data = JHMDBSubset(
    Path("data", "jhmdb"),
    partition=1,
    split="test",
    flatten=False,
    combined_transform=CropPerson(),
)

model = HRNetW32(N_JOINTS)
model.load_state_dict(torch.load(Path("weights", "hrnet_jhmdb_v9_1_best.pth")))
for gate in model.gates():
    gate.policy = ConstantThreshold(0.1)
model.eval()


def count_operations(video):
    model.counting()
    model.clear_counts()
    with torch.inference_mode():
        warm_up(video)
        model.clear_counts()
        model(video[-1].unsqueeze(dim=0))
        print(f'Ops: {model.total_counts()["conv2d_mac"]:.4g}', flush=True)


def profile_inference(video, json_path, verbose=False):
    model.no_counting()
    with torch.inference_mode():
        warm_up(video)
        with profile() as prof:
            model(video[-1].unsqueeze(dim=0))

    # Export results
    json_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(json_path))
    if verbose:
        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        print(table, flush=True)


def run_experiment(event_mode, sparse_mode, conv_mode, device, verbose=False):
    model.to(device)

    # Choose a video containing a moderate amount of motion.
    video = data[2][0][:4].to(device)

    model.eventful(mode=event_mode)
    model.sparse(mode=sparse_mode)
    model.set_modes(CountedConv2d, "conv_mode", conv_mode)

    event_str = "event" if event_mode else "conventional"
    sparse_str = "sparse" if sparse_mode else "non-sparse"
    print(
        f"{event_str.capitalize()},",
        f"{sparse_str},",
        f"{conv_mode} convolution,",
        f"{device.upper()}",
        flush=True,
    )
    json_path = Path(
        "outputs",
        "profile",
        "hrnet_jhmdb",
        f"{event_str}_{sparse_str}_{conv_mode}_{device}.json",
    )

    count_operations(video)
    time_inference(video)
    profile_inference(video, json_path, verbose=verbose)
    print()


def time_inference(video):
    model.no_counting()
    with torch.inference_mode():
        warm_up(video)
        t_1 = time.time()
        model(video[-1].unsqueeze(dim=0))
        torch.cuda.synchronize()
        t_2 = time.time()
        print(f"Wall time: {t_2 - t_1:.4f}", flush=True)


# Runs up to, but not including the last frame (the last frame is what
# we use for timing/profiling).
def warm_up(video):
    model.reset()
    for frame in video[:-1]:
        model(frame.unsqueeze(dim=0))


run_experiment(event_mode=False, sparse_mode=False, conv_mode="standard", device="cpu")
run_experiment(event_mode=False, sparse_mode=False, conv_mode="standard", device="cuda")

run_experiment(event_mode=True, sparse_mode=True, conv_mode="standard", device="cpu")
run_experiment(event_mode=True, sparse_mode=True, conv_mode="standard", device="cuda")

run_experiment(event_mode=False, sparse_mode=False, conv_mode="unrolled", device="cpu")
run_experiment(event_mode=False, sparse_mode=False, conv_mode="unrolled", device="cuda")

run_experiment(event_mode=True, sparse_mode=True, conv_mode="unrolled", device="cpu")
run_experiment(event_mode=True, sparse_mode=True, conv_mode="unrolled", device="cuda")

run_experiment(event_mode=False, sparse_mode=False, conv_mode="custom", device="cpu")

run_experiment(event_mode=True, sparse_mode=True, conv_mode="custom", device="cpu")
