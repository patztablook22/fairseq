#!/usr/bin/env python3
# Compute overlap of activations of enc-dec and decoder attention modules in TransformerModular
# TODO: support for more than single layer, single head Transformer
import sys
import numpy as np


def main():
    for line in sys.stdin:
        line = line.strip().split('\t')
        x = np.array([int(elem)for elem in line[0].split(',')])
        y = np.array([int(elem)for elem in line[1].split(',')])
        res = x + y
        print(','.join(res.astype(np.str)))


if __name__ == '__main__':
    main()
