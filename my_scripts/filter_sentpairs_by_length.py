#!/usr/bin/env python3

import sys
import argparse


def print_output(src, tgt, args):
    t_delim = args.token_delimiter
    if args.add_max_length_label:
        print("<{}>\t{}\t{}".format(
            args.max_length, t_delim.join(src), t_delim.join(tgt)))
    else:
        print("{}\t{}".format(t_delim.join(src), t_delim.join(tgt)))


def main(args) -> None:
    t_delim = args.token_delimiter
    s_delim = args.sentence_delimiter
    with open(args.input_file) as fh:
        for line in fh:
            line = line.strip().split(s_delim)
            if len(line) != 2:
                print("Does not contain a pair ({})".format(line), file=sys.stderr)
                continue
            src = line[0].split(t_delim)
            tgt = line[1].split(t_delim)
            #length = len(src) + len(tgt)

            if args.field == "both":
                if (
                    len(src) > args.min_length
                    and len(src) <= args.max_length
                    and len(tgt) > args.min_length
                    and len(tgt) <= args.max_length
                ):
                    print_output(src, tgt, args)

            else:
                if args.field == "src":
                    length = len(src)
                elif args.field == "tgt":
                    length = len(tgt)
                else:
                    raise ValueError("Unknown --field value: '{}'".format(args.field))
                if length > args.min_length and length <= args.max_length:
                    print_output(src, tgt, args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=str, default="/dev/stdin")
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--token-delimiter", default=" ")
    parser.add_argument("--sentence-delimiter", default="\t")
    parser.add_argument("--add-max-length-label", action="store_true")
    parser.add_argument("--field", type=str, default="src",
        help="which sentence should be used for measuring length (src, tgt, both)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
