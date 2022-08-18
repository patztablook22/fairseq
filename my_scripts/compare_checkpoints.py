#!/usr/bin/env python3
import argparse
import torch

from fairseq import checkpoint_utils


def main(args):
    state_1 = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint_1)
    ewc_1 = {n: p for n, p in state_1["model"].items() if "__" in n}
    state_1 = {n: p for n, p in state_1["model"].items() if "__" not in n}

    state_2 = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint_2)
    ewc_2 = {n: p for n, p in state_2["model"].items() if "__" in n}
    state_2 = {n: p for n, p in state_2["model"].items() if "__" not in n}
   
    for n, p in state_1.items():
        diff = torch.abs(state_1[n] - state_2[n])
        mean = diff.mean()
        std = diff.std()
        print("{}\t{}\t{}".format(n, mean, std))

    for n, p in ewc_1.items():
        diff = torch.abs(ewc_1[n] - ewc_2[n])
        mean = diff.mean()
        std = diff.std()
        print("{}\t{}\t{}".format(n, mean, std))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-1", type=str, required=True)
    parser.add_argument(
        "--checkpoint-2", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
