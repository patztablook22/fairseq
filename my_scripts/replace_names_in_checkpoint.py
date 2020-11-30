#!/usr/bin/env python3

import argparse
import torch


def main(args):
    ns_map = {}
    if args.namespace_map is not None:
        ns_map = {
            x.split("=")[0] : x.split("=")[1] for x in args.namespace_map.split(",")
        }
    ckpt_map = {}
    if args.ckpt_map is not None:
        ckpt_map = {
            x.split("=")[0] : x.split("=")[1] for x in args.ckpt_map.split(",")
        }

    ckpt = torch.load(args.checkpoint)
    ns = ckpt["args"]
    model = ckpt["model"]

    for k, v in ns_map.items():
        setattr(ns, k, v)
    for k, v in ckpt_map.items():
        model[v] = model.pop(k)

    ckpt["args"] = ns
    ckpt["model"] = model

    torch.save(ckpt, args.checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True)
    parser.add_argument(
        "--namespace-map", type=str, default=None)
    parser.add_argument(
        "--ckpt-map", type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
