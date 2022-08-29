#!/usr/bin/env python3
# Other suggestions:
# - remove trailing
import argparse

import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(src, tgt, m, ax):
    im = ax.imshow(m)

    THRESHOLD = 0.05

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(src)))
    ax.set_yticks(np.arange(len(tgt)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(src)
    ax.set_yticklabels(tgt)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    m[m < THRESHOLD] = 0.
    for i in range(len(src)):
        for j in range(len(tgt)):
            if m[j, i] < THRESHOLD:
                continue
            text = ax.text(
                i, j, "{:.2}".format(m[j, i]),
                ha="center", va="center", color="w", fontsize="xx-small")

    ax.set_title("")


def main(args):
    src = []
    tgt = []
    with open(args.text_file, "r") as fh:
        for i, line in enumerate(fh):
            if i == args.sentence_index:
                line = line.strip()
                src, tgt = line.split("\t")
                src = src.split(" ") + ["</s>"]
                tgt = ["<s>"] + tgt.split(" ")
    
    keys = ["encoder", "decoder", "encoder_decoder"]
    matrices_orig = np.load(args.attention_matrices)
    matrices = {k : matrices_orig[k] for k in keys}

    sent_idx = args.sentence_index
    if args.average_over_batch:
        sent_idx = 0
        for k in keys:
            matrices[k] = np.mean(matrices[k], 0, keepdims=True)

    n_heads = matrices["encoder_decoder"].shape[2]
    fig, axes = plt.subplots(3, n_heads + 1, figsize=(10 * n_heads, 30))
    for i in range(n_heads):
        #if len(src) != matrices["encoder"].shape[-1]:
        #    print("src {} - m.src {}".format(len(src), matrices["encoder"].shape[-1]))
        #if len(tgt) != matrices["decoder"].shape[-1]:
        #    print("tgt {} - m.tgt {}".format(len(tgt), matrices["decoder"].shape[-1]))
        plot_heatmap(
            src, src,
            matrices["encoder"][sent_idx, 0, i],
            axes[0, i])
        axes[0, i].set_title("EncSelf-{}".format(i))

        plot_heatmap(
            tgt, tgt,
            matrices["decoder"][sent_idx, 0, i],
            axes[1, i])
        axes[1, i].set_title("DecSelf-{}".format(i))      

        plot_heatmap(
            src, tgt,
            matrices["encoder_decoder"][sent_idx, 0, i],
            axes[2, i])
        axes[2, i].set_title("EncDec-{}".format(i))

    plt.show()
    plt.savefig("{}.png".format(args.save_prefix))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-file", type=str, required=True)
    parser.add_argument(
        "--attention-matrices", type=str, required=True)
    parser.add_argument(
        "--sentence-index", type=int, default=0)
    parser.add_argument(
        "--average-over-batch", action="store_true")
    parser.add_argument(
        "--save-prefix", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
