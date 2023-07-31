# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Any, Optional
from argparse import Namespace
from collections import OrderedDict
from omegaconf import OmegaConf

import numpy as np
from fairseq import utils
from fairseq.logging import metrics
from fairseq.data import (
    MultimodalLanguagePairDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from omegaconf import OmegaConf

from fairseq.cider.pyciderevalcap.ciderD.ciderD import CiderD


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_multimodal_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    patch_image_size=224,
    imagenet_default_mean_and_std=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    img_path = os.path.join(data_path, "{}.image.{}-{}".format(split, src, tgt))
    if indexed_dataset.dataset_exists(img_path, impl=dataset_impl):
        img_dataset = data_utils.load_indexed_dataset(
            img_path, None, dataset_impl
        )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return MultimodalLanguagePairDataset(
        img_dataset,
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        patch_image_size=patch_image_size,
        imagenet_default_mean_and_std=imagenet_default_mean_and_std,
    )


@dataclass
class MultimodalTranslationConfig(TranslationConfig):
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_tokenized_bleu: bool = field( 
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"} 
    ) 
    eval_bleu_remove_bpe: Optional[str] = field( 
        default=None, 
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_cider: bool = field(
        default=False, metadata={"help": "evaluation with CIDEr scores"}
    )
    eval_cider_cached_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"},
    )

    eval_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU or CIDER (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    imagenet_default_mean_and_std: Any = field(
        default=(0.5, 0.5, 0.5),
        metadata={"help": "Imagenet normalization values."}
    )

    patch_image_size: int = field(
        default=224,
        metadata={"help": "Size of the resized image input."}
    )

    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )


@register_task("multimodal_translation", dataclass=MultimodalTranslationConfig)
class MultimodalTranslationTask(TranslationTask):
    def __init__(self, cfg: MultimodalTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        if isinstance(cfg.imagenet_default_mean_and_std, str):
            self.imagenet_default_mean_and_std = eval(
                cfg.imagenet_default_mean_and_std
            )
        else:
            self.imagenet_default_mean_and_std = OmegaConf.to_container(
                self.cfg.imagenet_default_mean_and_std
            )
        self.patch_image_size = cfg.patch_image_size

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0

        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        # TODO: multimodal langpair dataset or separate?
        self.datasets[split] = load_multimodal_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            imagenet_default_mean_and_std=self.imagenet_default_mean_and_std,
            patch_image_size=self.patch_image_size,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, images=None, constraints=None):
        return MultimodalLanguagePairDataset(
            images,
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
            imagenet_default_mean_and_std=self.imagenet_default_mean_and_std,
            patch_image_size=self.patch_image_size,
        )


    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu or self.cfg.eval_cider:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
            if self.cfg.eval_cider:
                self.CiderD_scorer = CiderD(df=self.cfg.eval_cider_cached_tokens)
        if self.cfg.scst:
            raise ValueError("SCST generation is currently not supported...")
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )

        return model

    def _calculate_cider_scores(self, gen_res, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [gen_res[i].strip()]

        gts = OrderedDict()
        gt_res_ = [
            [gt_res[i][j].strip() for j in range(len(gt_res[i]))]
            for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[i]

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        _, scores = self.CiderD_scorer.compute_score(gts, res_)
        return scores

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if (
            self.cfg.eval_bleu
            or self.cfg.eval_cider 
            or self.cfg.eval_acc
            or self.cfg.eval_ter
        ):
            metrics = self._inference(self.sequence_generator, sample, model)

            if cfg.eval_bleu:
                logging_output["_bleu_sys_len"] = metrics["bleu"].sys_len
                logging_output["_bleu_ref_len"] = metrics["bleu"].ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = metrics["bleu"].counts[i]
                    logging_output["_bleu_totals_" + str(i)] = metrics["bleu"].totals[i]

            if self.cfg.eval_cider:
                logging_output["_cider_score_sum"] = metrics["cider"].sum()
                logging_output["_cider_cnt"] = metrics["cider"].size

            if self.cfg.eval_acc:
                logging_output["accuracy"] = metrics["accuracy"]

            if self.cfg.eval_ter:
                logging_output["ter"] = metrics["ter"].score
        
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if self.cfg.eval_bleu:
            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU
                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        import sacrebleu
                        # compatibility API for sacrebleu 1.x
                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

        if self.cfg.eval_cider:
            def compute_cider(meters):
                cider = meters["_cider_score_sum"].sum / meters["_cider_cnt"].sum
                cider = cider if isinstance(cider, float) else cider.item()
                return round(cider, 3)

            if sum_logs("_cider_cnt") > 0:
                metrics.log_scalar("_cider_score_sum", sum_logs("_cider_score_sum"))
                metrics.log_scalar("_cider_cnt", sum_logs("_cider_cnt"))
                metrics.log_derived("cider", compute_cider)

        if self.cfg.eval_acc:
            metrics.log_scalar('accuracy', sum_logs('accuracy'))

        if self.cfg.eval_ter:
            metrics.log_scalar('ter', sum_logs('ter'))

    def _inference(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return {
                "bleu" : sacrebleu.corpus_bleu(hyps, [refs], tokenize="none"),
                "cider" : self._calculate_cider_scores(hyps, refs),
                "ter" : sacrebleu.corpus_ter(hyps, [refs]),
                "accuracy" : sum([float(h == r) for h, r in zip(hyps, refs)]) / len(hyps)
            }
        return {
            "bleu" : sacrebleu.corpus_bleu(hyps, [refs]),
            "cider" : self._calculate_cider_scores(hyps, refs),
            "ter" : sacrebleu.corpus_ter(hyps, [refs]),
            "accuracy" : sum([float(h == r) for h, r in zip(hyps, refs)]) / len(hyps)
        }
