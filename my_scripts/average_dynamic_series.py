#!/usr/bin/env python3
# Other suggestions:
# - remove trailing
import argparse
import numpy as np

_INF=-1e9


def transform_values(x, transform_type):
    if transform_type == "exponential":
        return np.exp(x)
    return x


def main(args):
    res = None
    n = 0
    pad = args.pad
    with open(args.input_file, "r") as fh:
        for _, line in enumerate(fh):
            n += 1
            line = np.array([float(x) for x in line.strip().split(",")])
            line = transform_values(line, args.transform)
            if res is None:
                res = line
                continue
            # pad the shorter sequence
            if res.size < line.size:
                diff = line.size - res.size
                res = np.pad(
                    res, (0, diff), 'constant',
                    constant_values=(0, transform_values(pad, args.transform))
                )
            else:
                diff = res.size - line.size
                line = np.pad(
                    line, (0, diff), 'constant',
                    constant_values=(0, transform_values(pad, args.transform))
                )
            res += line
    print(','.join((res / n).astype(np.str)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=str, default="/dev/stdin")
    parser.add_argument(
        "--transform", type=str, default="exponential")
    parser.add_argument(
        "--pad", type=float, default=_INF)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
