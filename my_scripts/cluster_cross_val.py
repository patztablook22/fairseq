#!/usr/bin/env python3

import sys
import argparse
import random

import numpy as np

from sklearn.metrics.cluster import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score
)


def print_stats(x, y, quiet=True):
    ari = adjusted_rand_score(x, y)
    ami = adjusted_mutual_info_score(x, y)
    fms = fowlkes_mallows_score(x, y)

    if not quiet:
        print("ARI: {}".format(ari), file=sys.stderr)
        print("AMI: {}".format(ami), file=sys.stderr)
        print("FMS: {}".format(fms), file=sys.stderr)

    return ari, ami, fms


def compute_overlap(labels, vocabs): 
    same_cluster = {}
    for k in labels:
        x = np.array([labels[k]])
        same_cluster[k] = (x == x.transpose())

    res = np.zeros([len(vocabs), len(vocabs)])
    for i, a in enumerate(same_cluster.values()):
        for j, b in enumerate(same_cluster.values()):
            mask = (
                np.tril(np.ones(a.shape))
                * (np.diag(-np.ones(a.shape[0])) + 1)
            )
            res[i, j] = np.sum((a == b) * mask) / np.sum(mask)
    mask = (
        np.tril(np.ones(res.shape))
        * (np.diag(-np.ones(res.shape[0])) + 1)
    )
    res = res * mask
    mean = np.sum(res) / np.sum(mask)
    std = np.sum(((res - mean) ** 2) * mask) / np.sum(mask)
    return mean, std


def main(args):
    vocabs = {}
    labels = {}

    for f in args.input_files:
        voc = {}
        lab = []
        with open(f, 'r') as fh:
            for line in fh:
                line = line.strip()
                if line not in voc:
                    voc[line] = len(voc)
                lab.append(voc[line])
            vocabs[f] = voc
            labels[f] = lab

    res_ari = []
    res_ami = []
    res_fms = []
    res_overlap_mean, res_overlap_std = compute_overlap(labels, vocabs)
    for a in vocabs:
        for b in vocabs:
            if a == b:
                break

            if not args.quiet:
                print("Comparing... \n{}\n{}\n".format(a, b))
            if len(labels[a]) != len(labels[b]):
                print("Lengths do not match. Skipping...\n")
                continue

                print("Comparing B with A", file=sys.stderr)
            res = print_stats(labels[a], labels[b], quiet=args.quiet)

            res_ari.append(res[0])
            res_ami.append(res[1])
            res_fms.append(res[2])

    print("Compared {} pairs".format(len(res_ari)), file=sys.stderr)
    print("ARI: {} ({})".format(np.mean(res_ari), np.std(res_ari)))
    print("AMI: {} ({})".format(np.mean(res_ami), np.std(res_ami)))
    print("FMS: {} ({})".format(np.mean(res_fms), np.std(res_fms)))
    print("Overlap: {} ({})".format(res_overlap_mean, res_overlap_std))
 

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-files", type=str, nargs='+')
    parser.add_argument(
        "--quiet", action="store_true", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
