#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import numpy as np

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.models.transformer_modular import TransformerModularModel
from fairseq.data import encoders
from fairseq_cli.interactive import buffered_read, make_batches


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('extract_attention')


def main(args):
    ckpt = torch.load(args.checkpoint)
    ckpt['args'].path = args.checkpoint

    logger.info(ckpt['args'])

    utils.import_user_module(ckpt['args'])
    use_cuda = torch.cuda.is_available() and not ckpt['args'].cpu

    # Setup task
    task = tasks.setup_task(ckpt['args'])

    # Load model
    logger.info("loading model(s) from {}".format(ckpt['args'].path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        ckpt['args'].path.split(os.pathsep),
        arg_overrides=eval('{}'),
        task=task,
    )
    for model in models:
        if ckpt['args'].fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Initialize generator
    setattr(ckpt['args'], 'print_attention_weights', True)
    generator = task.build_generator(models, ckpt['args'])

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(ckpt['args'])
    bpe = encoders.build_bpe(ckpt['args'])

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    start_id = 0
    outputs = {
        "id": [],
        "encoder": [],
        "decoder": [],
        "enc_dec": [],
    }
    start_id = 0
    for inputs in buffered_read(args.input_file, 100):
        results = []
        with torch.no_grad():
            for batch in make_batches(inputs, ckpt['args'], task, max_positions, encode_fn):
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()

                sample = {
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": src_lengths,
                    },
                }

                translations = task.inference_step(generator, models, sample)
                for i, (idx, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    results.append((start_id + idx, hypos))

            for idx, hypos in sorted(results, key=lambda x: x[0]):
                outputs['id'].append(idx)
                attn_weights = hypos[0]["attn_weights"]

                for k in outputs:
                    if k == "id":
                        continue
                    outputs[k].append(
                        torch.stack(attn_weights[k]).cpu().numpy())

        start_id += len(inputs)

    for k in outputs:
        if k == "id":
            continue
        sizes = []
        for x in outputs[k]:
            sizes.append(x.shape)
        max_size = np.max(sizes, 0)
        for i, x in enumerate(outputs[k]):
            outputs[k][i] = np.pad(
                outputs[k][i],
                [[0, j] for j in (max_size - outputs[k][i].shape)],
                constant_values=0)
        outputs[k] = np.stack(outputs[k])
        
    np.savez(args.output_file, **outputs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True)
    parser.add_argument(
        "--input-file" type=str, required=True)
    parser.add_argument(
        "--output-file", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
