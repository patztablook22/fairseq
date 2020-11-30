#!/usr/bin/env python3
import argparse
import torch


def main(args):
    ckpt = torch.load(args.checkpoint)
    for k in ckpt['model'].keys():
        if args.varname_substr in k:
            print("{}\t{}".format(k, ckpt['model'][k].shape))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True)
    parser.add_argument(
        "--varname_substr", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
