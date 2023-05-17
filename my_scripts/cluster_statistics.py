#!/usr/bin/env python3

import argparse
import random
import numpy as np


def upper_tri_masking(x):
    m = x.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return x[mask]


def fleiss_kappa(x, N):
    p_e = x.sum(0) / (x.shape[0] * N)
    p_e = (p_e ** 2).sum()
    
    p = ((x ** 2).sum(1) - N) / (N ** 2 - N)
    p = p.sum() / x.shape[0]

    return (p - p_e) / (1 - p_e)
    

def main(args):
    vocabs = []
    labels = []

    for f in args.input_files:
        voc = {}
        lab = []
        with open(f, 'r') as fh:
            for line in fh:
                line = line.strip()
                if line not in voc:
                    voc[line] = len(voc)
                lab.append(voc[line])
            vocabs.append(voc)
            labels.append(np.array(lab))

    same_cluster = []
    for x in labels:
        x = np.expand_dims(x, 1)
        same_cluster.append(x == x.transpose())

    pair_counts = upper_tri_masking(np.sum(same_cluster, 0))
    pair_counts = np.stack([pair_counts, (len(same_cluster) - pair_counts)], 1)
    kappa = fleiss_kappa(pair_counts, len(same_cluster))
    print(kappa)

    contingency_tables = {}
    for i in range(len(labels)):
        for j in range(i):
            lab_i = labels[i]
            lab_j = labels[j]
            ct = np.zeros([
                np.unique(lab_i).shape[0],
                np.unique(lab_j).shape[0],
            ])
            for x, y in zip(lab_i, lab_j):
                ct[x, y] += 1
            contingency_tables["{}-{}".format(i, j)] = ct

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
