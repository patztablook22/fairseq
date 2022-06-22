# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


logger = logging.getLogger(__name__)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def compute_ewc(model, device, ewc_lambda, term_type="original"):
    ewc_sum = torch.tensor(0.).to(device)
    found_fisher = False  # Sanity check (mainly for debugging)

    logs = {}

    for n, p in model.named_parameters():
        n_orig = n
        n = n.replace('.', '__')

        found_fisher |= hasattr(model, '{}_mean'.format(n))
        if not found_fisher:
            continue

        mean = getattr(model, '{}_mean'.format(n), 0.)
        fisher = getattr(model, '{}_fisher'.format(n), 0.)
        if term_type == "original":
            ewc_sum += (fisher * (p - mean) ** 2).sum()
        elif term_type == "norm_1":
            ewc_sum += ((fisher / (fisher + 1)) * (p - mean) ** 2).sum()
        elif term_type == "norm_2":
            ewc_sum += ((fisher / (ewc_lambda * fisher + 1)) * (p - mean) ** 2).sum()
        else:
            raise ValueError("Wrong ewc_term_type")

        # logging
        fisher_argmax = fisher.argmax()
        logs["{}.max".format(n_orig)] = (p - mean).view(-1)[fisher_argmax]
        logs["{}__max".format(n)] = fisher.max()

        fisher_argmin = fisher.argmin()
        logs["{}.min".format(n_orig)] = (p - mean).view(-1)[fisher_argmin]
        logs["{}__min".format(n)] = fisher.min()

        fisher_argmedian = fisher.view(-1).argsort()[int(fisher.numel() / 2)]
        logs["{}.median".format(n_orig)] = (p - mean).view(-1)[fisher_argmedian]
        logs["{}__median".format(n)] = fisher.view(-1)[fisher_argmedian]

    if model.training and not found_fisher:
        logger.warning("Computing EWC although model does not "
                       "contain any weight consolidation info.")
    return ewc_sum, logs


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, ewc_lambda, ewc_term_type):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ewc_lambda = ewc_lambda
        self.ewc_term_type = ewc_term_type

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ewc-lambda', default=None, type=float, metavar='D',
                            help='lambda, weighting the EWC regularizer')
        parser.add_argument('--ewc-term-type', default="original", type=str, metavar='D',
                            help='how to compute the EWC regularizer loss')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, ewc_loss, ewc_logs = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ewc_loss': ewc_loss.data,
            'nll_loss': nll_loss.data,
            'ewc_logs': ewc_logs,
            'ewc_lambda': self.ewc_lambda,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        ewc_loss = torch.tensor(0.).to(loss.device)
        ewc_logs = {}
        if self.ewc_lambda is not None and self.ewc_lambda != 0.:
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

            # Compute the Regularizer loss and upscale it by sample size so
            # it's not batch-size independent. The gradients are later
            # normalized by sample_size in fairseq.trainer.
            ewc_loss, ewc_logs = compute_ewc(model, loss.device, self.ewc_lambda, term_type=self.ewc_term_type)
            ewc_loss *= sample_size
            loss += (self.ewc_lambda / 2) * ewc_loss

        return loss, nll_loss, ewc_loss, ewc_logs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ewc_loss_sum = sum(log.get('ewc_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ewc_loss', ewc_loss_sum / sample_size / math.log(2), 1, round=5)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        # EWC monitoring
        ewc_logs = logging_outputs[0].get('ewc_logs', {})
        for k in [k for k in ewc_logs.keys() if "__" not in k]:
            metrics.log_scalar(k, ewc_logs[k], 1, round=5)

        ewc_lambda = logging_outputs[0].get('ewc_lambda', 1.)
        metrics.log_scalar('ewc_2_loss', ewc_loss_sum * ewc_lambda / sample_size / math.log(2), 1, round=5)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
