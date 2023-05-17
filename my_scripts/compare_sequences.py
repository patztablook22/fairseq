#!/usr/bin/env python3
import sys
import sacrebleu
import pywer


def compute_accuracy(hyps, refs):
    acc = 0.0
    for h, r in zip(hyps, refs):
        if h == r:
            acc += 1.0
    return acc / float(len(refs))


def main():
    hypotheses = []
    references = []
    for line in sys.stdin:
        hyp, ref = line.split("\t")
        hypotheses.append(hyp.strip())
        references.append(ref.strip())
    wer = pywer.wer(references, hypotheses)
    ter = sacrebleu.compat.corpus_ter(hypotheses, [references]).score
    acc = compute_accuracy(hypotheses, references)

    print("WER: {:.3f}".format(wer))
    print("TER: {:.3f}".format(ter))
    print("ACC: {:.3f}".format(acc))


if __name__ == "__main__":
    main()
