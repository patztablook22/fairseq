#!/usr/bin/env python3

import argparse
import random
import numpy as np


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

    same_cluster = {}
    for k in labels:
        x = np.array([labels[k]])
        same_cluster[k] = (x == x.transpose())

    overlap = None
    for k in same_cluster:
        if same_cluster[k].shape[0] == 0:
            continue

        if overlap is None:
            overlap = same_cluster[k]
            continue
        overlap *= same_cluster[k]

    last_label = 0
    output_labels = []
    for i in range(overlap.shape[0]):
        label_i = None
        for j in range(i):
            if overlap[i][j]:
                label_i = output_labels[j]
                break
        if label_i is None:
            label_i = last_label
            last_label += 1
        output_labels.append(label_i)

    for lab in output_labels:
        print(lab)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-files", type=str, nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
