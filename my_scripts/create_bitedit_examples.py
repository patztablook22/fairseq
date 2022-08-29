#!/usr/bin/env python3
# Other suggestions:
# - remove trailing
import argparse

import numpy as np
from numpy.random import default_rng

ZERO_CHAR = 'a'
ONE_CHAR = 'b'

def binarize(n):
    res = []
    mapping = {
        "0" : ZERO_CHAR,
        "1" : ONE_CHAR
    }
    while n > 0:
        res = [str(n % 2)] + res
        n = n // 2
    res = [mapping[x] for x in res]
    return res


def create_id_example(x, tok=None, n=None):
    return (x, "-", "-")


def create_push_example(x, tok=ZERO_CHAR, n=1):
    # push
    return (x + [tok] * n, tok, str(n))


def create_pop_example(x, tok=None, n=1):
    # pop
    return (x[:-n], "-", str(n))


def create_unshift_example(x, tok=ZERO_CHAR, n=1):
    # unshift
    return ([tok] * n + x, tok, str(n))


def create_shift_example(x, tok=None, n=1):
    # shift
    return (x[n:], "-", str(n))


def create_remove_tokens_example(x, tok=ONE_CHAR, n=None):
    # remove
    return ([e for e in x if e != tok], tok, "-")


def create_reverse_example(x, tok=None, n=None):
    # reverse
    y = x.copy()
    y.reverse()
    return (y, "-", "-")


def create_duplicate_example(x, tok=None, n=None):
    # duplicate
    return (x + x, "-", "-")


def create_flip_example(x, tok=None, n=None):
    # flip
    mapping = {
        ZERO_CHAR : ONE_CHAR,
        ONE_CHAR : ZERO_CHAR,
    }
    return ([mapping[e] for e in x], "-", "-")


def create_flip_reverse_example(x, tok=None, n=None):
    # flip-reverse
    mapping = {
        ZERO_CHAR : ONE_CHAR,
        ONE_CHAR : ZERO_CHAR,
    }
    y = [mapping[e] for e in x]
    y.reverse()
    return (y, "-", "-")

TASK_MAP = {
    "id": create_id_example,
    "push": create_push_example,
    "pop": create_pop_example,
    "shift": create_shift_example,
    "unshift": create_unshift_example,
    "remove": create_remove_tokens_example,
    "reverse": create_reverse_example,
    "duplicate": create_duplicate_example,
    "flip": create_flip_example,
    "flip-reverse": create_flip_reverse_example
}


def main(args):
    if args.task not in TASK_MAP:
        raise ValueError("Undefined Task: '{}'".format(args.task))

    np.random.seed(args.seed)
    rng = default_rng(args.seed)
    x_all = rng.choice(
        2 ** args.max_n_bits - 2 ** args.min_n_bits,
        size=args.n_examples, replace=False)
    x_all += 2 ** args.min_n_bits
    for i, x in enumerate(x_all):
        x_bin = binarize(x)

        fn = TASK_MAP[args.task]
        if i % 2 == 0:
            y_bin, tok, n = fn(x_bin, ZERO_CHAR, args.n)
        else:
            y_bin, tok, n = fn(x_bin, ONE_CHAR, args.n)
        print("{} {} {}\t{}\t{}".format(
            args.task, tok, n, " ".join(x_bin), " ".join(y_bin)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=42)
    parser.add_argument(
        "--task", type=str, required=True)
    parser.add_argument(
        "--min-n-bits", type=int, default=1)
    parser.add_argument(
        "--max-n-bits", type=int, default=15)
    parser.add_argument(
        "--n", type=int, default=1)
    parser.add_argument(
        "--n-examples", type=int, default=3000)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
