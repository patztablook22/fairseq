#!/usr/bin/env python3
import argparse
import torch


def main(args):
    checkpoints = args.checkpoints.split(",")
    assert len(checkpoints) == 2

    ckpt1 = torch.load(checkpoints[0])
    ckpt2 = torch.load(checkpoints[1])

    for k in ckpt1['model'].keys():
        print("{}\t{}".format(k, (ckpt1['model'][k] == ckpt2['model'][k]).prod()))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
