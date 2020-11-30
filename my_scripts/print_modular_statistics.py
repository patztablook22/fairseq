#!/usr/bin/env python3
import sys
import argparse
import torch 


def main(args):
    subsets = {}
    modules = {}

    reindex_fn = lambda x: x
    if args.n_modules is not None and args.n_active is not None:
        print("n-modules and n-active set... using module reindexing...", file=sys.stderr)

        # We use torch to guarantee that the subset indexing will be same as with our model
        combos = torch.combinations(torch.arange(0, args.n_modules), args.n_active).numpy().tolist()
        reindex_fn = lambda x: combos[x]

    n_lines = 0
    with open(args.input_file, "r") as fh:
        for line in fh:
            n_lines += 1

            line = line.strip().split("\t")
            curr_modules = [int(x) for x in line[-1].split(",")]
            curr_modules = [y for x in curr_modules for y in reindex_fn(x)]
            curr_modules = [str(x) for x in curr_modules]

            s = ",".join(curr_modules)
            if s in subsets:
                subsets[s] += 1
            else:
                subsets[s] = 1

            for m in curr_modules:
                if m in modules:
                    modules[m] += 1
                else:
                    modules[m] = 1
    print("N-subsets\t{}".format(len(subsets.keys())))
    print("N-modules\t{}".format(len(modules.keys())))
    print("N-ratio\t{}".format(len(subsets.keys()) / float(len(combos))))

    for k, v in sorted(subsets.items(), key=lambda item: item[1], reverse=True):
        print("S-{}\t{}\t{}".format(k, v, (v / float(n_lines))))

    for k, v in sorted(modules.items(), key=lambda item: item[1], reverse=True):
        print("M-{}\t{}\t{}".format(k, v, (v / float(n_lines))))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=str, default="/dev/stdin")
    parser.add_argument(
        "--n-modules", type=int, default=None)
    parser.add_argument(
        "--n-active", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
