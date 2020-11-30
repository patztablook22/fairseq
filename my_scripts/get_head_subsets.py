#!/usr/bin/env python3
import argparse
import torch 


def main(args):
    res = []
       
    combos = torch.combinations(torch.arange(0, args.n_modules), args.n_active).numpy().tolist()
    for i, subset in enumerate(combos):
        if args.head_id in subset:
            res.append(str(i))

    print("{}".format(" ".join(res)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-id", type=int, required=True)
    parser.add_argument(
        "--n-modules", type=int, required=True)
    parser.add_argument(
        "--n-active", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
