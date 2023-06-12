# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import math
import torch
from torch.distributions.bernoulli import Bernoulli

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.modules import ModularCtrl


_EPS = 1e-9


@dataclass
class LabelSmoothedCrossEntropyCriterionModularConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    module_ctrl_max_temperature: float = field(
        default=10.,
        metadata={"help": "starting temperature for ctrl gumbel_sigmoid"},
    )
    module_ctrl_min_temperature: float = field(
        default=0.0625,
        metadata={
            "help": "minimum temperature for ctrl gumbel_sigmoid, default value taken"
            " from Ramesh et al. (2021), \"Zero-Shot Text-to-Image Generation\""
        },
    )
    module_ctrl_anneal_type: str = field(
        default="exponential",
        metadata={"help": "temperature annealing type [exponential, cosine]"},
    )
    module_ctrl_anneal_rate: float = field(
        default=1e-6,
        metadata={
            "help": "temperature anneal rate (exponential annealing) for controller\'s"
            " gumbel_sigmoid"
        },
    )
    module_ctrl_cosine_reset_decay: float = field(
        default=0.95,
        metadata={"help": "TODO"},
    )
    module_ctrl_cosine_reset_every_n_steps: int = field(
        default=1,
        metadata={"help": "TODO"},
    )
    module_kl_div_regularizer_ratio: float = field(
        default=.5,
        metadata={
            "help": "a regularization ratio of modules that should be unmasked"
            " by the controller (used in kl_div regularizer)"
        },
    )
    module_kl_div_regularizer_weight: float = field(
        default=0.,
        metadata={"help": "weighting of the kl_div regularization term"},
    )
    module_budget_regularizer_ratio: float = field(
        default=.5,
        metadata={
            "help": "a regularization ratio of modules that should be unmasked"
            " by the controller (used in budget regularizer)"
        },
    )
    module_budget_regularizer_weight: float = field(
        default=0.,
        metadata={"help": "weighting of the budget regularization term"},
    )
    vertical_penalty_weight: float = field(
        default=0.,
        metadata={"help": "TODO"},
    )


def selection_entropy(controllers):
    entropies = [
        ModularCtrl._mask_output_probs(
            Bernoulli(logits=ctrl["logits"]).entropy(), ctrl["padding_mask"]
        ) for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(
            torch.ones_like(ctrl["logits"]), ctrl["padding_mask"]
        ) for ctrl in controllers if ctrl is not None
    ]

    if entropies:
        return torch.cat(entropies, axis=1).sum() / torch.cat(n_all, axis=1).sum()
    return torch.tensor(0.)


def batch_selection_entropy(controllers):
    def layer_entropy(layer_ctrl):
        probs = ModularCtrl._mask_output_probs(
            Bernoulli(logits=layer_ctrl["logits"]).probs, layer_ctrl["padding_mask"])
        n_all = ModularCtrl._mask_output_probs(
            torch.ones_like(layer_ctrl["logits"]), layer_ctrl["padding_mask"])
        probs = probs.sum(0) / n_all.sum(0)
        probs = torch.stack([probs, 1 - probs], -1)
        return (-probs * torch.log(probs + _EPS)).sum(-1)

    entropies = [
        ModularCtrl._mask_output_probs(
            layer_entropy(ctrl),
            (ctrl["padding_mask"].all(0) if ctrl["padding_mask"] is not None else None)
        ) for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(
            torch.ones_like(ctrl["logits"][0]),
            (ctrl["padding_mask"].all(0) if ctrl["padding_mask"] is not None else None)
        ) for ctrl in controllers if ctrl is not None
    ]

    if entropies:
        return torch.cat(entropies, axis=0).sum() / torch.cat(n_all, axis=0).sum()
    return torch.tensor(0.)


def compute_masked_ratio(controllers, reduce=True):
    n_masked = [
        ModularCtrl._mask_output_probs(ctrl["mask"], ctrl["padding_mask"]).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(torch.ones_like(ctrl["mask"]), ctrl["padding_mask"]).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    if n_all:
        n_masked = torch.stack(n_masked, 1).sum(1)
        n_all = torch.stack(n_all, 1).sum(1)
        res = n_masked / (n_all + _EPS)

        assert res.dim() == 1

        if reduce:
            return res.sum()
        return res
    return torch.tensor(0.)


def compute_masked_budget(controllers, mask_budget, reduce=True):
    assert 0. <= mask_budget <= 1.
    n_masked = [
        ModularCtrl._mask_output_probs(ctrl["mask"], ctrl["padding_mask"]).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    n_all = [
        ModularCtrl._mask_output_probs(torch.ones_like(ctrl["mask"]), ctrl["padding_mask"]).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    if n_all:
        n_masked = torch.stack(n_masked, 1).sum(1)
        n_all = torch.stack(n_all, 1).sum(1)
        res = n_masked / (n_all + _EPS)

        assert res.dim() == 1
        res = torch.sqrt((res - mask_budget)**2 + _EPS)  # EPS for numerical stability

        if reduce:
            return res.sum()
        return res
    return torch.tensor(0.)


def compute_kl_div(controllers, q, reduce=True):
    res = []
    for ctrl in controllers:
        if ctrl is None:
            continue
        probs = torch.sigmoid(ctrl["logits"])
        kl_div = -(q * torch.log(probs + _EPS) + (1 - q) * torch.log(1 - probs + _EPS))
        res.append(ModularCtrl._mask_output_probs(kl_div, ctrl["padding_mask"]))

    if res:
        res = torch.cat(res, dim=1)
        res = res.sum([-2, -1])

        assert res.dim() == 1

        if reduce:
            return res.sum()
        return res
    return torch.tensor(0.)


def compute_vertical_penalty(controllers, reduce=True):
    n_masked = [
        ModularCtrl._mask_output_probs(ctrl["mask"], ctrl["padding_mask"]).sum([-2, -1])
        for ctrl in controllers if ctrl is not None
    ]
    if n_masked:
        res = torch.stack(n_masked, 1).sum(1)
        assert res.dim() == 1

        if reduce:
            return res.sum()
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


@register_criterion(
    "modular_label_smoothed_cross_entropy",
    dataclass=LabelSmoothedCrossEntropyCriterionModularConfig,
)
class ModularLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        module_ctrl_max_temperature,
        module_ctrl_min_temperature,
        module_ctrl_anneal_type,
        module_ctrl_anneal_rate,
        module_ctrl_cosine_reset_decay,
        module_ctrl_cosine_reset_every_n_steps,
        module_kl_div_regularizer_ratio,
        module_kl_div_regularizer_weight,
        module_budget_regularizer_ratio,
        module_budget_regularizer_weight,
        vertical_penalty_weight,
    ):
        # TODO: move the annealing-related parameters to a single JSON-like parameter
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy,
        )
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
        self.vertical_penalty_weight = vertical_penalty_weight

        self.temp = self.module_ctrl_max_temperature

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.training:
            self.update_temperature(sample)

        net_output = model(**sample['net_input'], ctrl_temperature=self.temp)
        loss, nll_loss, regularizer_stats = self.compute_loss(
            model, net_output, sample, reduce=reduce
        )
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "module_kl_div": regularizer_stats["kl_div"].data,
            "module_mask_budget": regularizer_stats["mask_budget"].data,
            "module_mask_ratio": regularizer_stats["mask_ratio"].data,
            "vertical_penalty": regularizer_stats["vertical_penalty"].data,
            "sel_entropy": regularizer_stats["sel_entropy"].data,
            "batch_entropy": regularizer_stats["batch_entropy"].data,
            "temperature": self.temp,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def update_temperature(self, sample):
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

    def get_regularizer_values(self, model, net_output, reduce=True):
        ctrl_outputs = net_output[1]["ctrl_outputs"]
        module_ctrl_outputs = [
            ctrl_out for key in ctrl_outputs for ctrl_out in ctrl_outputs[key]
            if "vert" not in key and ctrl_out is not None
        ]
        vert_ctrl_outputs = [
            ctrl_out for key in ctrl_outputs for ctrl_out in ctrl_outputs[key]
            if "vert" in key and ctrl_out is not None
        ]

        return {
            "sel_entropy": selection_entropy(module_ctrl_outputs),
            "batch_entropy": batch_selection_entropy(module_ctrl_outputs),
            "kl_div": compute_kl_div(
                module_ctrl_outputs, self.module_kl_div_regularizer_ratio, reduce
            ),
            "mask_budget": compute_masked_budget(
                module_ctrl_outputs, self.module_budget_regularizer_ratio, reduce
            ),
            "mask_ratio": compute_masked_ratio(module_ctrl_outputs, reduce),
            "vertical_penalty": compute_vertical_penalty(vert_ctrl_outputs, reduce),
        }

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss, nll_loss = super().compute_loss(model, net_output, sample)
        regularizer_stats = self.get_regularizer_values(
            model,
            net_output,
            reduce=reduce,
        )

        loss += self.vertical_penalty_weight * regularizer_stats["vertical_penalty"]
        loss += self.module_kl_div_regularizer_weight * regularizer_stats["kl_div"]
        loss += self.module_budget_regularizer_weight * regularizer_stats["mask_budget"]


        return loss, nll_loss, regularizer_stats

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)

        kl_div_sum = sum(log.get("module_kl_div", 0) for log in logging_outputs)
        mask_budget_sum = sum(log.get("module_mask_budget", 0) for log in logging_outputs)
        mask_ratio_sum = sum(log.get("module_mask_ratio", 0) for log in logging_outputs)
        vertical_penalty_sum = sum(log.get("vertical_penalty", 0) for log in logging_outputs)

        sel_entropy = sum(log.get("sel_entropy", 0) for log in logging_outputs)
        batch_entropy = sum(log.get("batch_entropy", 0) for log in logging_outputs)
        temp = sum(log.get("temperature", 0) for log in logging_outputs)

        metrics.log_scalar(
            "module_KL_div", kl_div_sum / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "module_mask_budget", mask_budget_sum / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "module_mask_ratio", mask_ratio_sum / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "vertical_penalty", vertical_penalty_sum / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "average_depth", vertical_penalty_sum / ntokens, ntokens, round=3
        )

        metrics.log_scalar("sel_entropy", sel_entropy, 1, round=3)
        metrics.log_scalar("batch_entropy", batch_entropy, 1, round=3)
        metrics.log_scalar("temperature", temp, 1, round=3)
