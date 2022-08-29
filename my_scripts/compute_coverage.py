#!/usr/bin/env python3
import argparse

import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(m, ax):
    im = ax.imshow(m)

    THRESHOLD = 0.05

    depth = m.shape[0]
    width = m.shape[1]

    # We want to show all ticks...
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(depth))
    # ... and label them with the respective list entries
    #ax.set_xticklabels(range(width))
    #ax.set_yticklabels(range(height))

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor")

    print(m.shape)
    # Loop over data dimensions and create text annotations.
    m[m < THRESHOLD] = 0.
    for i in range(width):
        for j in range(depth):
            if m[j, i] < THRESHOLD:
                continue
            text = ax.text(
                i, j, "{:.2}".format(m[j, i]),
                ha="center", va="center", color="w", fontsize="xx-small")

    ax.set_title("")


def line2numpy(line):
    res = []

    line = line.split(';')
    for layer in line:
        layer = layer.split(':')[1]
        layer = np.array([[int(x) for x in elem] for elem in layer.split(',')])
        res.append(layer)
    # Shape: (len, n_layers, n_heads)
    return np.stack(res, axis=1)


def main(args):
    all_masks = []
    with open(args.input_file, "r") as fh:
        for line in fh:
            all_masks.append(line2numpy(line.strip()))
    all_masks = np.concatenate(all_masks, 0)
    if args.plot_heatmap:
        fig, ax = plt.subplots(1, 1, figsize=(30, 30))
        plot_heatmap(all_masks.mean(0), ax)
        plt.show()
    print('Unmasked percentage:\t{:.3f}'.format(all_masks.mean()))
    print('{}'.format(all_masks.mean(0)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=str, default="/dev/stdin")
    parser.add_argument(
        "--plot-heatmap", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
