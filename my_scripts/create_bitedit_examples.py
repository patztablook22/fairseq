#!/usr/bin/env python3
# Other suggestions:
# - remove trailing
import argparse

import numpy as np

def generate_x(min_length, max_length):
    length = np.random.randint(low=min_length, high=max_length)
    return np.random.choice(2, size=length)

def create_id_example(x, tok=None, n=None):
    return (x, "-", "-")

def create_push_example(x, tok, n=1):
    return (np.concatenate([x, [tok] * n]), tok, str(n))

def create_pop_example(x, tok, n=1):
    return (x[:-n], "-", str(n))

def create_unshift_example(x, tok, n=1):
    return (np.concatenate([tok] * n, x), tok, str(n))

def create_shift_example(x, tok, n=1):
    return (x[n:], "-", str(n))

def create_remove_tokens_example(x, tok, n=None):
    return (x[x != tok], tok, "-")

def create_reverse_example(x, tok=None, n=None):
    return (x[::-1], "-", "-")

def create_duplicate_example(x, tok=None, n=None):
    return (np.concatenate([x, x]), "-", "-")

def create_flip_example(x, tok=None, n=None):
    return (1 - x, "-", "-")

def create_flip_reverse_example(x, tok=None, n=None):
    return (1 - x[::-1], "-", "-")

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
    for i in range(args.n_examples):
        x = generate_x(args.min_n_bits, args.max_n_bits)
        fn = TASK_MAP[args.task]
        y, tok, n = fn(x, i % 2, args.n)
        mapping = np.array([args.zero_char, args.one_char])
        x_c = mapping[x]
        y_c = mapping[y]
        print(f"{args.task} {tok} {n}\t{' '.join(x_c)}\t{' '.join(y_c)}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--min-n-bits", type=int, default=1)
    parser.add_argument("--max-n-bits", type=int, default=15)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--n-examples", type=int, default=3000)
    parser.add_argument("--zero-char", type=str, default='a')
    parser.add_argument("--one-char", type=str, default='b')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
