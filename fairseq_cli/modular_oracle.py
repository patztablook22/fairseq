#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders, LanguagePairDataset
from fairseq.logging import progress_bar
from fairseq.criterions.label_smoothed_cross_entropy_modular import (
    LabelSmoothedCrossEntropyModularCriterion
)


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')


Translation = namedtuple('TranslationModular', 'src_str hypos pos_scores alignments')


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
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield batch


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    ckpt = torch.load(args.path)
    task = tasks.setup_task(ckpt['args'])
    model = task.build_model(ckpt['args'])

    criterion = task.build_criterion(ckpt['args'])
    assert isinstance(criterion, LabelSmoothedCrossEntropyModularCriterion)
    criterion.eval()

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

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

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for sample in make_batches(inputs, args, task, max_positions, encode_fn):
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            with torch.no_grad():
                assert len(models) == 1

                selections = []
                for model in models:
                    if (
                            args.fixed_encoder_selection is not None
                            or args.fixed_decoder_selection is not None
                        ):
                        selection = {
                            'encoder' : None,
                            'decoder' : None,
                        }
                        if args.fixed_encoder_selection is not None:
                            sel = torch.tensor(eval(args.fixed_encoder_selection))
                            selection['encoder'] = sel.repeat(sample['id'].size(0), 1)
                        if args.fixed_decoder_selection is not None:
                            sel = torch.tensor(eval(args.fixed_decoder_selection))
                            selection['decoder'] = sel.repeat(sample['id'].size(0), 1)
                    else:
                        # 1. Compute outputs for every ctrl selection
                        sampled_outputs = criterion.sample_outputs(model, sample, random_samples=False)

                        # 2. Take selection with the lowest loss (given true predictions)
                        selection = criterion.compute_best_selection(model, sampled_outputs, sample)
                    selections.append(selection)

                # 3. Use the best selection to predict output in the inference mode
                translations = generator.generate(models, sample, selections)

                for i, (id, hypos) in enumerate(zip(sample['id'].tolist(), translations)):
                    src_tokens_i = utils.strip_pad(sample['net_input']['src_tokens'][i], tgt_dict.pad())
                    results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print('H-{}\t{}\t{}'.format(id, score, hypo_str))
                # detokenized hypothesis
                print('D-{}\t{}\t{}'.format(id, score, detok_hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        # convert from base e to base 2
                        hypo['positional_scores'].div_(math.log(2)).tolist(),
                    ))
                ))
                if args.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(
                        id,
                        alignment_str
                    ))
                if 'enc_selection' in hypo:
                    print('Menc-{}\t{}'.format(id, hypo['enc_selection']))
                if 'dec_selection' in hypo:
                    print('Mdec-{}\t{}'.format(id, hypo['dec_selection']))
                if args.print_attn_confidence:
                    print('C-{}\t{}'.format(id, hypo['enc_self_attn_conf']))

        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
