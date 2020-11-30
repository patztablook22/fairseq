#!/usr/bin/env python3

import argparse
import sys


ID_TAG = "0_id"
SEL_TAG = "example (encoder) selection"

def main(args):
    results = []
    scores = []
    headcount = []

    epoch = {}
    with open(args.logfile, "r") as fh:
        for line in fh:
            line = [x.strip() for x in line.split("|")]
            if len(line) < 3:
                continue
            if line[2] == "fairseq.tasks.translation_modular":
                if "example " not in line[3]:
                    continue
                item = [x.strip() for x in line[3].split(":")]
                if item[0] not in epoch:
                    epoch[item[0]] = [item[1]]
                else:
                    epoch[item[0]].append(item[1])
                if item[0] == "example hypothesis":
                    if ID_TAG not in epoch:
                        epoch[ID_TAG] = ["0"]
                    else:
                        epoch[ID_TAG].append(str(len(epoch["0_id"])))
            if line[2] == "valid":
                scores.append(line[12].split(" ")[1])
                headcount.append(line[17].split(" ")[1:])
                results.append(epoch)
                epoch = {}
    if args.print_head_counts:
        for r in results:
            heads = {}
            for sel in r[SEL_TAG]:
                for h in sel.split(","):
                    if h not in heads:
                        heads[h] = 1
                    else:
                        heads[h] += 1
            print(" ".join(["{}-{}".format(k, v) for k, v in heads.items()]))
        sys.exit()

    if args.print_subset_counts:
        for r in results:
            subsets = {}
            for sel in r[SEL_TAG]:
                if sel not in subsets:
                    subsets[sel] = 1
                else:
                    subsets[sel] += 1
            print(" ".join(["{}-{}".format(k, v) for k, v in subsets.items()]))
        sys.exit()

    for r, s, h in zip(results, scores, headcount):
        keys = sorted(list(r.keys()))
        for i in range(len(r[keys[0]])):
             print("\t".join([r[k][i] for k in keys]))
        print("VALID\t{}\t{}".format(s, h))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logfile", type=str, required=True)
    parser.add_argument("--print-head-counts", action="store_true")
    parser.add_argument("--print-subset-counts", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
