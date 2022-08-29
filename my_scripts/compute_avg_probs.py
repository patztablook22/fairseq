#!/usr/bin/env python3
import sys
import numpy as np


def line2numpy(line):
    res = []

    line = line.split(';')
    for layer in line:
        layer = layer.split(':')[1]
        layer = np.array([[float(x) for x in elem.split('#')] for elem in layer.split(',')])
        res.append(layer)
    # Shape: (len, n_layers, n_heads)
    return np.stack(res, axis=1)


def main():
    all_masks = []
    for line in sys.stdin:
        all_masks.append(line2numpy(line.strip()))
    all_masks = np.concatenate(all_masks, 0)
    print('Mean probability:\t{:.3f} ({:.3f})'.format(all_masks.mean(), all_masks.std()))
    print('Head probabilities (mean):\t{}'.format(all_masks.mean(0)))
    print('Head probabilities (std):\t{}'.format(all_masks.std(0)))


if __name__ == '__main__':
    main()
