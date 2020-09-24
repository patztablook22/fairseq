# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


_EPS = 1e-9


def selection_entropy(controllers):
    return torch.stack([
        ctrl.entropy().mean() for ctrl in controllers
        if ctrl is not None], axis=0).mean()


def batch_selection_entropy(controllers):
    def layer_entropy(layer_ctrl):
        probs = layer_ctrl.probs.mean(0)
        return (-probs * torch.log(probs + _EPS)).sum(-1)

    return torch.stack([
        layer_entropy(ctrl) for ctrl in controllers
        if ctrl is not None], axis=0).mean()


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


def compute_sel_lprobs(ctrl_outputs):
    def compute_single(ctrl_out):
        return ctrl_out.ctrl.log_prob(
            ctrl_out.ctrl_prediction).sum(-1)

    sel_lprobs = torch.stack([
        compute_single(out) for out in ctrl_outputs if out is not None
    ], axis=0).sum(0)

    return sel_lprobs


@register_criterion('label_smoothed_cross_entropy_modular')
class LabelSmoothedCrossEntropyModularCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, ctrl_alpha, e_step_size, m_steps):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ctrl_alpha = ctrl_alpha
        
        self.e_step_size = e_step_size
        self.m_steps = m_steps

        self.step_id = 1  # indicates when to apply e-step
        assert self.m_steps > 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ctrl-alpha', default=1., type=float, metavar='A',
                            help='ctrl selection loss weight')
        parser.add_argument('--e-step-size', default=10, type=int, metavar='N',
                            help='number of samples per expectation step '
                                 '(negative value means compute across all '
                                 'ctrl configurations)')
        parser.add_argument('--m-steps', default=10, type=int, metavar='N',
                            help='number of maximization steps between the e-steps')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # Expectation step (do not save gradients)
        if self.step_id % (self.m_steps + 1) == 0:
            if self.e_step_size != 0:
                sampled_outputs = self.sample_outputs(
                    model, sample,
                    sample_size=self.e_step_size)
                self.update_best_ctrl_selection(model, sampled_outputs, sample)
        self.step_id = (self.step_id + 1) % (self.m_steps + 1)

        net_out = model(**sample['net_input'], data_indices=sample['id'], mode="m_step")
        loss, nll_loss, ctrl_loss, sel_entropy, batch_entropy = self.compute_loss(
            model, net_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ctrl_loss': ctrl_loss.data,
            'sel_entropy': sel_entropy.data,
            'batch_entropy': batch_entropy.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def sample_outputs(self, model, sample, sample_size=-1):
        with torch.no_grad():
            bsz = sample['id'].size(0)
            sample_selections = model.list_all_selections()
            random.shuffle(sample_selections)
            outputs = []

            # negative sample_size is a shortcut for using all possible selections
            if sample_size >= 0:
                sample_selections = sample_selections[:sample_size]

                # We run the m-step first, to get estimated outputs of
                # the current best selections
                outputs.append(
                    model(
                        **sample['net_input'],
                        data_indices=sample['id'],
                        mode='m_step',
                    )
                )

            for selection in sample_selections:
                for key in [k for k, v in selection.items() if v is not None]:
                    selection[key] = selection[key].repeat(bsz, 1)
                outputs.append(
                    model(
                        **sample['net_input'],
                        mode='e_step',
                        fixed_selection=selection,
                    )
                )
            return outputs

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        bsz = lprobs.size(0)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        ctrl_outputs = [
            ctrl_out for ctrl_out in net_output[1]['ctrl_output'].values()
        ]
        sel_lprobs = compute_sel_lprobs(ctrl_outputs)
        sel_loss = -sel_lprobs

        sel_entropy = selection_entropy([c.ctrl for c in ctrl_outputs if c is not None])
        batch_entropy = batch_selection_entropy([c.ctrl for c in ctrl_outputs if c is not None])

        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx,
            reduce=False,
        )
        loss = loss.view(bsz, -1).sum(1)

        if reduce:
            loss = loss.sum()
            nll_loss = nll_loss.sum()
            sel_loss = sel_loss.sum()
        loss = loss + self.ctrl_alpha * sel_loss
        return loss, nll_loss, sel_loss, sel_entropy, batch_entropy

    def update_best_ctrl_selection(self, model, net_outputs, sample):
        selection = self.compute_best_selection(model, net_outputs, sample)
        model.update_best_ctrl_selection(selection, sample['id'])

    def compute_best_selection(self, model, net_outputs, sample):
        lprobs = [model.get_normalized_probs(out, log_probs=True) for out in net_outputs]
        bsz = lprobs[0].size(0)
        lprobs = [lp.view(-1, lp.size(-1)) for lp in lprobs]

        # We compute the target loglikelihoods via nll_loss function
        # We want non-smoothed probability for the best_prediction estimation
        target = model.get_targets(sample, net_outputs[0]).view(-1, 1)
        nll_losses = []
        for lp in lprobs:
            _, nll_loss = label_smoothed_nll_loss(
                lp, target, self.eps, ignore_index=self.padding_idx,
                reduce=False)
            nll_losses.append(nll_loss.view(bsz, -1).sum(1))

        # shape(bsz, n_samples)
        nll_losses = torch.stack(nll_losses, axis=1)
        loglikelihoods = -nll_losses

        ctrl_outputs = [
            net_out[1]['ctrl_output'] for net_out in net_outputs
        ]
        sel_lprobs = torch.stack([
            compute_sel_lprobs(
                [val for val in ctrl_out.values()]
            ) for ctrl_out in ctrl_outputs
        ], axis=1)

        joint_lprobs = loglikelihoods + sel_lprobs
        best_sel_indices = joint_lprobs.max(1)[1].detach()
        best_sel_indices = [list(range(bsz)), best_sel_indices]

        selection = {
            "encoder": torch.stack(
                [
                    net_out[1]['ctrl_output']['encoder'].selection
                    for net_out in net_outputs
                ], axis=1)[best_sel_indices],
            "decoder": None
        }
        if net_outputs[0][1]['ctrl_output']['decoder'] is not None:
            selection["decoder"] = torch.stack(
                [
                    net_out[1]['ctrl_output']['decoder'].selection
                    for net_out in net_outputs
                ], axis=1)[best_sel_indices]
        return selection

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ctrl_loss_sum = sum(log.get('ctrl_loss', 0) for log in logging_outputs)
        sel_entropy = sum(log.get('sel_entropy', 0) for log in logging_outputs)
        batch_entropy = sum(log.get('batch_entropy', 0) for log in logging_outputs)
        ctrl_entropy_ratio = sel_entropy / batch_entropy
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('ctrl_loss', ctrl_loss_sum / sample_size / math.log(2), nsentences, round=3)
        metrics.log_scalar('sel_entropy', sel_entropy, 1, round=3)
        metrics.log_scalar('batch_entropy', batch_entropy, 1, round=3)
        metrics.log_scalar('ctrl_entropy_ratio', ctrl_entropy_ratio, 1, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
