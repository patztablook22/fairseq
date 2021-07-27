# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import torch

from torch.distributions.bernoulli import Bernoulli

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


_EPS = 1e-9


def compute_masked_ratio(controllers):
    # TODO: use this when computing loss for additional statistics
    # TODO: debug this
    n_masked = sum([ctrl.mask.sum([0, 2]) for ctrl in controllers if ctrl is not None])
    n_all = sum([ctrl.mask[0].numel() for ctrl in controllers if ctrl is not None])
    if n_all:
        res = n_masked / n_all
        assert 0. <= res.mean() <= 1.
        return res
    return None


def compute_kl_div(controllers, q):
    probs = [
        torch.sigmoid(ctrl.logits)
        for ctrl in controllers if ctrl is not None
    ]
    if not probs:
        return torch.tensor(0.)
    probs = torch.cat(probs, dim=1)

    res = -(q * torch.log(probs + _EPS) + (1 - q) * torch.log(1 - probs + _EPS)).sum([-2, -1])
    return res


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


# TODO: move these two methods somewhere else (into the criterion class?)
def exponential_annealing(temp, step, anneal_rate, min_temp):
    return max(temp * math.exp(-anneal_rate * step), min_temp)


def cosine_annealing(temp, step, min_temp, max_temp, epoch, reset=False):
    if reset:
        return min_temp + 0.5 * (max_temp - min_temp) * (1 + math.cos(math.pi * epoch))
    return temp + 0.5 * (max_temp - min_temp) * (1 + math.cos(math.pi * epoch))


@register_criterion('label_smoothed_cross_entropy_modular')
class LabelSmoothedCrossEntropyModularCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 module_ctrl_max_temperature,
                 module_ctrl_min_temperature,
                 module_ctrl_anneal_rate,
                 module_ctrl_anneal_type,
                 module_coverage_regularizer_ratio,
                 module_coverage_regularizer_weight):
        # TODO: parameter renaming?
        super().__init__(task, sentence_avg, label_smoothing)
        self.module_ctrl_max_temperature = module_ctrl_max_temperature
        self.module_ctrl_min_temperature = module_ctrl_min_temperature
        self.module_ctrl_anneal_rate = module_ctrl_anneal_rate
        self.module_ctrl_anneal_type = module_ctrl_anneal_type
        self.module_coverage_regularizer_ratio = module_coverage_regularizer_ratio
        self.module_coverage_regularizer_weight = module_coverage_regularizer_weight

        self.temp = self.module_ctrl_max_temperature
        self.prev_epoch = 0

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(
            LabelSmoothedCrossEntropyModularCriterion,
            LabelSmoothedCrossEntropyModularCriterion).add_args(parser)
        parser.add_argument('--module-ctrl-max-temperature', type=float, default=1.,
                            help='starting temperature for ctrl gumbel_sigmoid')
        parser.add_argument('--module-ctrl-min-temperature', type=float, default=(0.0625),
                            help='minimum temperature for ctrl gumbel_sigmoid, default value taken '
                                 'from Ramesh et al. (2021), "Zero-Shot Text-to-Image Generation"')
        parser.add_argument('--module-ctrl-anneal-rate', type=float, default=1e-6,
                            help='temperature anneal rate for ctrl gumbel_sigmoid')
        parser.add_argument('--module-ctrl-anneal-type', type=str, default='exponential',
                            help='temperature annealing type')
        parser.add_argument('--module-coverage-regularizer-ratio', type=float, default=.5,
                            help='a regularization ratio of modules that should be chosen by the controller')
        parser.add_argument('--module-coverage-regularizer-weight', type=float, default=1.,
                            help='weighting of the regularization term')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.training:
            step = sample["update_num"]
            epoch = sample["epoch_num"]
            if self.module_ctrl_anneal_type == 'exponential':
                self.temp = exponential_annealing(
                    self.temp, step, self.module_ctrl_anneal_rate,
                    self.module_ctrl_min_temperature)
            elif self.module_ctrl_anneal_type == 'cosine':
                self.temp = cosine_annealing(
                    self.temp, step, self.module_ctrl_min_temperature,
                    self.module_ctrl_max_temperature, epoch, self.prev_epoch != epoch)
                self.prev_epoch = epoch
            elif self.module_ctrl_anneal_type == 'constant':
                self.temp = self.module_ctrl_max_temperature
            else:
                raise ValueError('Unknown module annealing type: {}'.format(self.module_ctrl_anneal_type))

        net_output = model(**sample['net_input'], ctrl_temperature=self.temp)
        loss, nll_loss, kl_div = self.compute_loss(
            model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'module_kl_div': kl_div.data,
            'temperature': self.temp,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        bsz = lprobs.size(0)

        # Flatten the controller outputs structure for computing relevant statistics (kl_div, etc.)
        ctrl_outputs = net_output[1]['ctrl_outputs']
        ctrl_outputs = [
            ctrl_out for key in ctrl_outputs for ctrl_out in ctrl_outputs[key]
            if ctrl_out is not None
        ]

        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx,
            reduce=False,
        )
        loss = loss.view(bsz, -1).sum(1)

        kl_div = compute_kl_div(ctrl_outputs, self.module_coverage_regularizer_ratio)

        if reduce:
            loss = loss.sum()
            nll_loss = nll_loss.sum()
            kl_div = kl_div.sum()
        loss += self.module_coverage_regularizer_weight * kl_div

        return loss, nll_loss, kl_div

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        kl_div_sum = sum(log.get('module_kl_div', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        temp = sum(log.get('temperature', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('module_KL_div', kl_div_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('temperature', temp, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
