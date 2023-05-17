#!/usr/bin/env python3
# Other suggestions:
# - remove trailing
import argparse
import sys


def compute_id(x, tok=None, n=None):
    # id (copy)
    return x


def compute_push(x, tok, n=1):
    # push
    return x + [tok] * n


def compute_pop(x, tok=None, n=1):
    # pop
    return x[:-n]


def compute_unshift(x, tok, n=1):
    # unshift
    return [tok] * n + x


def compute_shift(x, tok=None, n=1):
    # shift
    return x[n:]


def compute_remove(x, tok, n=None):
    # remove
    return [e for e in x if e != tok]


def compute_reverse(x, tok=None, n=None):
    # reverse
    y = x.copy()
    y.reverse()
    return y


def compute_duplicate(x, tok=None, n=None):
    # duplicate
    return x + x


def compute_flip(x, flip_mapping, tok=None, n=None):
    # flip
    return [flip_mapping[e] for e in x]


def compute_flip_reverse(x, flip_mapping, tok=None, n=None):
    # flip-reverse
    y = [mapping[e] for e in x]
    y.reverse()
    return y


def main(args):
    flip_mapping = {
        args.zero_char : args.one_char,
        args.one_char : args.zero_char,
    }

    task_map = {
        "id" : compute_id,
        "push" : compute_push,
        "pop" : compute_pop,
        "shift" : compute_shift,
        "unshift" : compute_unshift,
        "remove" : compute_remove,
        "reverse" : compute_reverse,
        "duplicate" : compute_duplicate,
        "flip" : lambda x, tok, n=None: compute_flip(x, flip_mapping, tok, n),
        "flip-reverse" : lambda x, tok, n=None: compute_flip_reverse(x, flip_mapping, tok, n),
    }

    if args.task not in task_map:
        raise ValueError("Undefined Task: '{}'".format(args.task))

    map_fn = task_map[args.task]
    with open(args.input_file, "r") as fh:
        for i, line in enumerate(fh):
            line = line.strip().split(" ")
            #print("{} {}".format(i, " ".join(line), sys.stderr))
            print(" ".join(map_fn(line, args.token)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=str, default="/dev/stdin")
    parser.add_argument(
        "--task", type=str, required=True)
    parser.add_argument(
        "--zero-char", type=str, default='a')
    parser.add_argument(
        "--one-char", type=str, default='b')
    parser.add_argument(
        "--token", type=str, default='a')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
