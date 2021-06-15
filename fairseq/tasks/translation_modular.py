# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging

import numpy as np

from fairseq import metrics, utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

_EPS = 1e-9


logger = logging.getLogger(__name__)


@register_task('translation_modular')
class TranslationModularTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.
    (Using modular transformer.)

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        for coder in ['encoder', 'decoder']:
            label = coder + '_ctrl_probs'
            if any([(label not in log) for log in logging_outputs]):
                continue

            def compute_batch_entropy(meters):
                probs = meters.get('_' + label)
                if probs is not None:
                    return (-probs * np.log(probs + _EPS)).mean()

            probs = torch.cat([log.get(label) for log in logging_outputs], 0).cpu()
            sel_entropy = (-probs * torch.log(probs + _EPS)).mean()

            probs_mean = probs.mean(0)
            batch_entropy = (-probs_mean * torch.log(probs_mean + _EPS)).mean()

            metrics.log_scalar('_' + label, np.array(probs.mean(0)))
            metrics.log_scalar(coder + '_ctrl_sel_entropy', np.array(sel_entropy))
            metrics.log_scalar(coder + '_ctrl_batch_entropy', np.array(batch_entropy))

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs, enc_mod_mask, dec_mod_mask = [], [], [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
            if 'enc_module_mask' in gen_out[i][0]:
                enc_mod_mask.append(gen_out[i][0]['enc_module_mask'])
            if 'dec_module_mask' in gen_out[i][0]:
                dec_mod_mask.append(gen_out[i][0]['dec_module_mask'])
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
            if enc_mod_mask:
                logger.info('example (encoder) selection: ' + enc_mod_mask[0])
            if dec_mod_mask:
                logger.info('example (decoder) selection: ' + dec_mod_mask[0])
        if self.args.eval_tokenized_bleu:
            return (
                sacrebleu.corpus_bleu(hyps, [refs], tokenize='none'),
                sacrebleu.corpus_ter(hyps, [refs]),
                sum([float(h == r) for h, r in zip(hyps, refs)]) / len(hyps)
            )
        else:
            return (
                sacrebleu.corpus_bleu(hyps, [refs]),
                sacrebleu.corpus_ter(hyps, [refs]),
                sum([float(h == r) for h, r in zip(hyps, refs)]) / len(hyps)
            )
