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
from fairseq.modules import ModularCtrl


_EPS = 1e-9


def selection_entropy(controllers):
    entropies = [
        ModularCtrl._mask_output_probs(
            Bernoulli(logits=ctrl.logits).entropy(), ctrl.padding_mask
        ) for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(
            torch.ones_like(ctrl.logits), ctrl.padding_mask
        ) for ctrl in controllers if ctrl is not None
    ]

    if entropies:
        return torch.cat(entropies, axis=1).sum() / torch.cat(n_all, axis=1).sum()
    return torch.tensor(0.)


def batch_selection_entropy(controllers):
    def layer_entropy(layer_ctrl):
        probs = ModularCtrl._mask_output_probs(
            Bernoulli(logits=layer_ctrl.logits).probs, layer_ctrl.padding_mask)
        n_all = ModularCtrl._mask_output_probs(
            torch.ones_like(layer_ctrl.logits), layer_ctrl.padding_mask)
        probs = probs.sum(0) / n_all.sum(0)
        probs = torch.stack([probs, 1 - probs], -1)
        return (-probs * torch.log(probs + _EPS)).sum(-1)

    entropies = [
        ModularCtrl._mask_output_probs(
            layer_entropy(ctrl),
            (ctrl.padding_mask.all(0) if ctrl.padding_mask is not None else None)
        ) for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(
            torch.ones_like(ctrl.logits[0]),
            (ctrl.padding_mask.all(0) if ctrl.padding_mask is not None else None)
        ) for ctrl in controllers if ctrl is not None
    ]

    if entropies:
        return torch.cat(entropies, axis=0).sum() / torch.cat(n_all, axis=0).sum()
    return torch.tensor(0.)


def compute_masked_ratio(controllers):
    n_masked = [
        ModularCtrl._mask_output_probs(ctrl.mask, ctrl.padding_mask).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(torch.ones_like(ctrl.mask), ctrl.padding_mask).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    if n_all:
        n_masked = torch.stack(n_masked, 1).sum(1)
        n_all = torch.stack(n_all, 1).sum(1)
        res = n_masked / (n_all + _EPS)

        assert res.dim() == 1
        return res
    return torch.tensor(0.)


def compute_masked_budget(controllers, mask_budget):
    assert 0. <= mask_budget <= 1.
    n_masked = [
        ModularCtrl._mask_output_probs(ctrl.mask, ctrl.padding_mask).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(torch.ones_like(ctrl.mask), ctrl.padding_mask).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    if n_all:
        n_masked = torch.stack(n_masked, 1).sum(1)
        n_all = torch.stack(n_all, 1).sum(1)
        res = n_masked / (n_all + _EPS)

        assert res.dim() == 1
        return torch.sqrt((res - mask_budget)**2 + _EPS)  # EPS for numerical stability
    return torch.tensor(0.)


def compute_kl_div(controllers, q):
    res = []
    for ctrl in controllers:
        if ctrl is None:
            continue
        probs = torch.sigmoid(ctrl.logits)
        kl_div = -(q * torch.log(probs + _EPS) + (1 - q) * torch.log(1 - probs + _EPS))
        res.append(ModularCtrl._mask_output_probs(kl_div, ctrl.padding_mask))

    if res:
        res = torch.cat(res, dim=1)
        res = res.sum([-2, -1])

        assert res.dim() == 1
        return res
    return torch.tensor(0.)


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


def exponential_annealing(temp, step, anneal_rate, min_temp):
    return max(temp * math.exp(-anneal_rate * step), min_temp)


def cosine_annealing(step, max_steps, min_temp, max_temp):
    assert max_temp > min_temp
    return min_temp + 0.5 * (max_temp - min_temp) * (1 + math.cos(math.pi * step / max_steps))


@register_criterion('label_smoothed_cross_entropy_modular')
class LabelSmoothedCrossEntropyModularCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 module_ctrl_max_temperature,
                 module_ctrl_min_temperature,
                 module_ctrl_anneal_type,
                 module_ctrl_anneal_rate,
                 module_ctrl_cosine_reset_decay,
                 module_ctrl_cosine_reset_every_n_steps,
                 module_kl_div_regularizer_ratio,
                 module_kl_div_regularizer_weight,
                 module_budget_regularizer_ratio,
                 module_budget_regularizer_weight):
        # TODO: move the annealing-related parameters to a single JSON-like parameter
        super().__init__(task, sentence_avg, label_smoothing)
        self.module_ctrl_max_temperature = module_ctrl_max_temperature
        self.module_ctrl_min_temperature = module_ctrl_min_temperature

        self.module_ctrl_anneal_type = module_ctrl_anneal_type
        self.module_ctrl_anneal_rate = module_ctrl_anneal_rate
        self.module_ctrl_cosine_reset_decay = module_ctrl_cosine_reset_decay
        self.module_ctrl_cosine_reset_every_n_steps = module_ctrl_cosine_reset_every_n_steps

        self.module_kl_div_regularizer_ratio = module_kl_div_regularizer_ratio
        self.module_kl_div_regularizer_weight = module_kl_div_regularizer_weight
        self.module_budget_regularizer_ratio = module_budget_regularizer_ratio
        self.module_budget_regularizer_weight = module_budget_regularizer_weight

        self.temp = self.module_ctrl_max_temperature

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(
            LabelSmoothedCrossEntropyModularCriterion,
            LabelSmoothedCrossEntropyModularCriterion).add_args(parser)
        parser.add_argument('--module-ctrl-max-temperature', type=float, default=10.,
                            help='starting temperature for ctrl gumbel_sigmoid')
        parser.add_argument('--module-ctrl-min-temperature', type=float, default=(0.0625),
                            help='minimum temperature for ctrl gumbel_sigmoid, default value taken '
                                 'from Ramesh et al. (2021), "Zero-Shot Text-to-Image Generation"')
        parser.add_argument('--module-ctrl-anneal-type', type=str, default='exponential',
                            help='temperature annealing type [exponential, cosine]')
        parser.add_argument('--module-ctrl-anneal-rate', type=float, default=1e-6,
                            help='temperature anneal rate (exponential annealing) for controller\'s gumbel_sigmoid')
        parser.add_argument('--module-ctrl-cosine-reset-decay', type=float, default=0.95,
                            help='TODO')
        parser.add_argument('--module-ctrl-cosine-reset-every-n-steps', type=float, default=1.,
                            help='TODO')
        parser.add_argument('--module-kl-div-regularizer-ratio', type=float, default=.5,
                            help='a regularization ratio of modules that should be unmasked by the controller (used in kl_div regularizer)')
        parser.add_argument('--module-kl-div-regularizer-weight', type=float, default=0.,
                            help='weighting of the kl_div regularization term')
        parser.add_argument('--module-budget-regularizer-ratio', type=float, default=.5,
                            help='a regularization ratio of modules that should be unmasked by the controller (used in budget regularizer)')
        parser.add_argument('--module-budget-regularizer-weight', type=float, default=0.,
                            help='weighting of the budget regularization term')
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
            if self.module_ctrl_anneal_type == 'exponential':
                self.temp = exponential_annealing(
                    self.temp,
                    step,
                    self.module_ctrl_anneal_rate,
                    self.module_ctrl_min_temperature)
            elif self.module_ctrl_anneal_type == 'cosine':
                step = (step + 1) % self.module_ctrl_cosine_reset_every_n_steps
                if step == 0:
                    self.module_ctrl_max_temperature *= self.module_ctrl_cosine_reset_decay
                self.temp = cosine_annealing(
                    step,
                    self.module_ctrl_cosine_reset_every_n_steps,
                    self.module_ctrl_min_temperature,
                    self.module_ctrl_max_temperature)
            elif self.module_ctrl_anneal_type == 'constant':
                self.temp = self.module_ctrl_max_temperature
            else:
                raise ValueError('Unknown module annealing type: {}'.format(self.module_ctrl_anneal_type))

        net_output = model(**sample['net_input'], ctrl_temperature=self.temp)
        loss, nll_loss, module_stats = self.compute_loss(
            model, net_output, sample, reduce=reduce)

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'module_kl_div': module_stats["kl_div"].data,
            'module_mask_budget': module_stats["mask_budget"].data,
            'module_mask_ratio': module_stats["mask_ratio"].data,
            'sel_entropy' : module_stats["sel_entropy"].data,
            'batch_entropy' : module_stats["batch_entropy"].data,
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

        module_stats = {}

        # Flatten the controller outputs structure for computing relevant statistics (kl_div, etc.)
        ctrl_outputs = net_output[1]['ctrl_outputs']
        ctrl_outputs = [
            ctrl_out for key in ctrl_outputs for ctrl_out in ctrl_outputs[key]
            if ctrl_out is not None
        ]

        module_stats["sel_entropy"] = selection_entropy(ctrl_outputs)
        module_stats["batch_entropy"] = batch_selection_entropy(ctrl_outputs)

        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx,
            reduce=False,
        )
        loss = loss.view(bsz, -1).sum(1)

        kl_div = compute_kl_div(ctrl_outputs, self.module_kl_div_regularizer_ratio)
        mask_budget = compute_masked_budget(ctrl_outputs, self.module_budget_regularizer_ratio)
        mask_ratio = compute_masked_ratio(ctrl_outputs)

        if reduce:
            loss = loss.sum()
            nll_loss = nll_loss.sum()
            kl_div = kl_div.sum()
            mask_ratio = mask_ratio.sum()
            mask_budget = mask_budget.sum()
        loss += self.module_kl_div_regularizer_weight * kl_div
        loss += self.module_budget_regularizer_weight * mask_budget

        module_stats["kl_div"] = kl_div
        module_stats["mask_ratio"] = mask_ratio
        module_stats["mask_budget"] = mask_budget

        return loss, nll_loss, module_stats

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        logging_len = len(logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        kl_div_sum = sum(log.get('module_kl_div', 0) for log in logging_outputs)
        mask_budget_sum = sum(log.get('module_mask_budget', 0) for log in logging_outputs)
        mask_ratio_sum = sum(log.get('module_mask_ratio', 0) for log in logging_outputs)
        sel_entropy = sum(log.get('sel_entropy', 0) for log in logging_outputs)
        batch_entropy = sum(log.get('batch_entropy', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        temp = sum(log.get('temperature', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('module_KL_div', kl_div_sum / ntokens, ntokens, round=3)
        metrics.log_scalar('module_mask_budget', mask_budget_sum / nsentences, nsentences, round=3)
        metrics.log_scalar('module_mask_ratio', mask_ratio_sum / nsentences, nsentences, round=3)
        metrics.log_scalar('sel_entropy', sel_entropy, 1, round=3)
        metrics.log_scalar('batch_entropy', batch_entropy, 1, round=3)
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
