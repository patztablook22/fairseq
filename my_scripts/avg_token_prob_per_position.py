#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np


INF=1e-9


def main(args):
    all_lprobs = []
    max_len = 0
    with open(args.input_file, "r") as fh:
        for line in fh:
            lprobs = [float(x) for x in line.strip().split(",")]
            if max_len < len(lprobs):
                max_len = len(lprobs)
            all_lprobs.append(lprobs)

    all_lprobs = [x  + [INF] * (max_len - len(x)) for x in all_lprobs]
    average = np.mean(all_lprobs, 0)
    if args.use_probs:
        average = np.exp(average)
    print(",".join(average.astype(np.str)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=str, default="/dev/stdin")
    parser.add_argument(
        "--use-probs", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
