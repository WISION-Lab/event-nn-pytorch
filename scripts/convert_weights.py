#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch


def main(args):
    # Parse correspondence patterns from the text file.
    patterns = parse_patterns(args.pattern_file)

    # Read input weights into a dictionary.
    if Path(args.in_file).suffix == ".pth":
        in_weights = torch.load(args.in_file)
    else:
        in_weights = np.load(args.in_file)
        in_weights = {
            key: torch.from_numpy(weight) for key, weight in in_weights.items()
        }
    in_weights = dict(in_weights)

    # Rename and permute weights.
    n_remapped = 0
    out_weights = {}
    for in_key, weight in in_weights.items():
        out_key = in_key
        for regex, replacement, permutation in patterns:
            out_key, n_matches = regex.subn(replacement, out_key)
            if n_matches > 0:
                n_remapped += 1
                if permutation is not None:
                    weight = weight.permute(permutation)
                if args.verbose:
                    print(f"{in_key}  ==>  {out_key}")
        out_weights[out_key] = weight
    print(f"Remapped {n_remapped}/{len(in_weights)} weights.")
    torch.save(out_weights, args.out_file)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("in_file", help="input .pth or .npz file")
    parser.add_argument(
        "pattern_file",
        help=".txt file containing regex patterns and shape permutations",
    )
    parser.add_argument("out_file", help=".pth file where the output should be saved")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print detailed output"
    )
    return parser.parse_args()


def parse_patterns(pattern_file):
    patterns = []
    line_1 = None
    line_2 = None
    with open(pattern_file, "r") as text:
        for line in text:
            line = line.strip()
            if line == "":
                line_1 = None
                line_2 = None
            elif line_2 is not None:
                regex = re.compile(line_1)
                if line == "-":
                    permutation = None
                else:
                    permutation = tuple(int(s) for s in line.split(","))
                patterns.append((regex, line_2, permutation))
            elif line_1 is not None:
                line_2 = line
            else:
                line_1 = line
    return patterns


if __name__ == "__main__":
    main(parse_args())
