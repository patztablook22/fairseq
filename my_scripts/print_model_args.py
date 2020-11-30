#!/usr/bin/env python3

import argparse
import torch


def main(args):
    ckpt = torch.load(args.checkpoint)
    ns = ckpt["args"]
    
    for line in ns.__dict__.items():
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
