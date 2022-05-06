#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Estimate models Fisher Information (based on gradients computed over a give sample).
Store the information to the model
"""

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch
from torch import autograd

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders, LanguagePairDataset
from fairseq.trainer import Trainer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')


Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for line in h:
            src_str, tgt_str = line.strip().split('\t')
            buffer.append((src_str, tgt_str))
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    src_tokens = []
    tgt_tokens = []
    for src_str, tgt_str in lines:
        src_tokens.append(
            task.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
        )
        tgt_tokens.append(
            task.target_dictionary.encode_line(
                encode_fn(tgt_str), add_if_not_exist=False
            ).long()
        )
    src_lengths = [t.numel() for t in src_tokens]
    tgt_lengths = [t.numel() for t in tgt_tokens]

    dataset = LanguagePairDataset(
        src_tokens, src_lengths, task.source_dictionary,
        tgt_tokens, tgt_lengths, task.target_dictionary
    )
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=True)
    for batch in itr:
        yield batch


def main(args):
    utils.import_user_module(args)

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # We need to load the model similarly to training-restore
    # We will then update themodel with FI values and save it for later training resume
    train_args = torch.load(args.path)['args']

    task = tasks.setup_task(args)
    model = task.build_model(train_args)
    criterion = task.build_criterion(train_args)

    trainer = Trainer(train_args, task, model, criterion, None)
    # HACK: set the restore_file arg which is not defined for Generation mode parser
    train_args.restore_file = args.path
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(train_args, trainer)
    # TODO: avoid loading the training dataset OR compute FI by subsampling
    #       from training dataset

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions()]
    )

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    
    n_batches = 0
    model.zero_grad()
    for inputs in buffered_read(args.input, args.buffer_size):
        for sample in make_batches(inputs, args, task, max_positions, encode_fn):
            n_batches += 1
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            # Perform train step to get loss to compute gradients
            model.train()
            loss, _, _ = criterion(model, sample)

            # Accumulate gradients
            loss.backward()

    # Average gradients and compute fisher diagonal
    fisher_diagonals = [(p.grad / n_batches) ** 2 for n, p in model.named_parameters()]

    param_names = [
        n.replace('.', '__') for n, p in model.named_parameters()
    ]
    fisher = {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    # Consolidate
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        model.register_buffer('{}_mean'.format(n), p.data.clone())
        model.register_buffer('{}_fisher'.format(n), fisher[n].data.clone())

    # Save
    # HACK: We want to make epoch_itr.end_of_epoch method callable without raising error
    #       for the save_checkpoint method.
    epoch_itr.epoch -= 1
    epoch_itr.next_epoch_itr()

    checkpoint_utils.save_checkpoint(train_args, trainer, epoch_itr, None)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
