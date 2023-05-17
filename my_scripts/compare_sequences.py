#!/usr/bin/env python3
import argparse
import sys
import sacrebleu
import pywer


def compute_accuracy(hyps, refs):
    acc = 0.0
    for h, r in zip(hyps, refs):
        if h == r:
            acc += 1.0
    return acc * 100 / float(len(refs))


def main(args):
    hypotheses = []
    references = []
    with open(args.input_file, "r") as fh:
        for line in fh:
            hyp, ref = line.split("\t")
            hypotheses.append(hyp.strip())
            references.append(ref.strip())

    if args.per_example_acc:
        for hyp, ref in zip(hypotheses, references):
            print(compute_accuracy([hyp], [ref]))
    else:
        acc = compute_accuracy(hypotheses, references)
        print("ACC: {:.3f}".format(acc))

    if not args.acc_only:
        wer = pywer.wer(references, hypotheses)
        ter = sacrebleu.compat.corpus_ter(hypotheses, [references]).score

        print("WER: {:.3f}".format(wer))
        print("TER: {:.3f}".format(ter))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=str, default="/dev/stdin")
    parser.add_argument(
        "--acc-only", action="store_true")
    parser.add_argument(
        "--per-example-acc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
