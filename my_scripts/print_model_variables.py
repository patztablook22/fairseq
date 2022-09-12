#!/usr/bin/env python3
import argparse
import torch


def main(args):
    ckpt = torch.load(args.checkpoint)
    for k in ckpt['model'].keys():
        minimum = ckpt['model'][k].min()
        maximum = ckpt['model'][k].max()
        mean = ckpt['model'][k].mean()
        #var = ckpt['model'][k].var()
        std = ckpt['model'][k].std()
        median = ckpt['model'][k].median()
        print("{}\t{}\t{}\t{}\t{}\t{}".format(k, minimum, maximum, mean, std, median))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
