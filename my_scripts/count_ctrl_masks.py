#!/usr/bin/env python3
import argparse
import numpy as np


def parse_ctrl_string(ctrl_outputs_str):
    ctrl_outputs = {}
    # See fairseq.sequence_generator.SequenceGeneratorWithModuleMask for syntax
    for ctrl_str in ctrl_outputs_str.split(";"):
        layer_num_str, ctrl_str = ctrl_str.split(":")

        res = []
        for mask in ctrl_str.split(","):
            res.append([x for x in mask])
        res = np.array(res).astype(np.int32).reshape(len(res), -1)
        ctrl_outputs[layer_num_str] = res
    return ctrl_outputs


def update_counts(ctrl_outputs,
                  counts,
                  inputs=None,
                  count_isolated_modules=True):
    for k, v in ctrl_outputs.items():
        counts = update("n_all", v.shape[0], counts)
        
        # Use dummy inputs if not provided by the user
        if inputs is None:
            inputs = ["XXX" for _ in range(v.shape[0])]
        for x_in in inputs:
            counts = update("{}".format(x_in), 1, counts)

        for pos, (ctrl_out, x_in) in enumerate(zip(v, inputs)):
            if count_isolated_modules:
                for i in range(ctrl_out.shape[0]):
                    counts = update("{}:{}:n_all".format(k, i), ctrl_out[i], counts)
                    counts = update("{}:{}:{}".format(k, i, x_in), ctrl_out[i], counts)
                    counts = update("{}:{}:{}:{}".format(k, i, x_in, pos), ctrl_out[i], counts)
            else:
                counts = update("{}:{}:n_all".format(k, "".join(ctrl_out.astype(np.str))), 1, counts)
                counts = update("{}:{}:{}".format(k, "".join(ctrl_out.astype(np.str)), x_in), 1, counts)
                counts = update("{}:{}:{}:{}".format(k, "".join(ctrl_out.astype(np.str)), x_in, pos), 1, counts)
    return counts

        

def update(k, v, d):
    if k not in d:
        d[k] = v
    else:
        d[k] += v
    return d


def main(args):
    counts = {}
    
    with open(args.input_file, "r") as fh:
        for line in fh:
            line = line.strip().split("\t")
            assert len(line) == 1 or len(line) == 2

            ctrl_outputs = parse_ctrl_string(line[0])

            inputs = None
            if len(line) == 2:
                inputs = line[1].split(" ")

            counts = update_counts(ctrl_outputs, counts, inputs, args.count_isolated_modules)
    for k, v in counts.items():
        print("{}\t{}".format(k, v))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=str, default="/dev/stdin",
        help="File containing tab-separated 'head-selection' and (option) "
             "inputs for the ctrl")
    parser.add_argument(
        "--count-isolated-modules", action="store_true",
        help="count the module selection in isolation (not in the context "
             "of other modules)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
