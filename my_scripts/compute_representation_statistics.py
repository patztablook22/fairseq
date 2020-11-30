#!/usr/bin/env python2.7
from __future__ import print_function

import argparse
import numpy as np

from metrics import (
    cosine_similarity,
    nearest_neighbors,
    masked_mean,
    mean_nearest_neighbor_overlap,
    pearson,
)


KEYS = ['enc_ctrl_feat', 'dec_ctrl_feat']
LABELS = ['enc_ctrl_pred', 'dec_ctrl_pred']
PROJECTIONS = ['enc_ctrl_proj', 'dec_ctrl_proj']


def main(args):
    outputs = []
    for r in args.representations:
        outputs.append(np.load(r, allow_pickle=True))

    for idx, key in enumerate(KEYS):
        # Compute mNN
        res = []
        for i, _ in enumerate(outputs):
            for j in range(i + 1, len(outputs)):
                assert outputs[i][key].shape == outputs[j][key].shape
                res.append(
                    mean_nearest_neighbor_overlap(
                        outputs[i][key], outputs[j][key], args.num_neighbors))
        print("m{}NN-{} {:.3f} {:.3f}".format(
            args.num_neighbors, key, np.mean(res), np.var(res)))

        # Compute intra/inter cluster similarities...
        # ...and pearson correlation between ctrl featurs and output matrices
        res = {
            "c_intra": [],
            "c_inter": [],
            "pearson": [],
        }
        for i, _ in enumerate(outputs):
            mask_intra = np.equal(
                np.expand_dims(outputs[i][LABELS[idx]], 0),
                np.expand_dims(outputs[i][LABELS[idx]], 1)).astype(np.float32).prod(-1)
            np.fill_diagonal(mask_intra, 0)
            mask_inter = 1 - mask_intra
            np.fill_diagonal(mask_inter, 0)

            sim = cosine_similarity(outputs[i][key])
            res["c_intra"].append(masked_mean(sim, mask_intra))
            res["c_inter"].append(masked_mean(sim, mask_inter))

            projections = outputs[i][PROJECTIONS[idx]][outputs[i][LABELS[idx]]]
            projections = projections.mean(1)
            res["pearson"].append(pearson(outputs[i][key], projections, mask_inter))

        print("c-intra-{} {:.3f} {:.3f}".format(
            key, np.mean(res["c_intra"]), np.var(res["c_intra"])))
        print("c-inter-{} {:.3f} {:.3f}".format(
            key, np.mean(res["c_inter"]), np.var(res["c_inter"])))
        print("pearson-{} {:.3f} {:.3f}".format(
            key, np.mean(res["pearson"]), np.var(res["pearson"])))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representations", type=str, nargs='+')
    parser.add_argument(
        "--num_neighbors", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
