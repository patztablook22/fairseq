# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


EPS = 1e-9


def selection_entropy(net_out):
    controllers = net_out[1]['controllers']
    return torch.stack([
        ctrl.entropy().mean() for ctrl in controllers
        if ctrl is not None], axis=0).mean()


def batch_selection_entropy(net_out):
    def layer_entropy(layer_ctrl):
        probs = layer_ctrl.probs.mean(0)
        return (-probs * torch.log(probs + EPS)).sum(-1)

    controllers = net_out[1]['controllers']
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
        if pad_mask.any():
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


def compute_sel_lprobs(net_out, mask_repetitions=True):
    selections = net_out[1]['selections']
    controllers = net_out[1]['controllers']
    sel_lprobs = torch.stack([
        ctrl.log_prob(selections[i]) for i, ctrl in enumerate(controllers)
        if ctrl is not None
    ], axis=-1).sum(-1)

    return sel_lprobs.sum(-1)


@register_criterion('label_smoothed_cross_entropy_modular')
class LabelSmoothedCrossEntropyModularCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        
        self.e_steps = args.e_steps
        assert self.e_steps >= 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # Expectation step (do not save gradients)
        with torch.no_grad():
            sampled_outputs = []
            model.reset_best_selection()
            for i in range(self.e_steps):
                net_out = model(**sample['net_input'], mode="e_step")
                sampled_outputs.append(net_out)
            if self.e_steps > 0:
                self.update_best_selection(model, sampled_outputs, sample)

        net_out = model(**sample['net_input'], mode="m_step")
        loss, nll_loss, ctrl_loss, sel_entropy, batch_entropy = self.compute_loss(
            model, net_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ctrl_loss': utils.item(ctrl_loss.data) if reduce else ctrl_loss.data,
            'sel_entropy': utils.item(sel_entropy.data),
            'batch_entropy': utils.item(batch_entropy.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        bsz = lprobs.size(0)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        sel_lprobs = compute_sel_lprobs(net_output)
        sel_loss = -sel_lprobs

        sel_entropy = selection_entropy(net_output)
        batch_entropy = batch_selection_entropy(net_output)

        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx,
            reduce=False,
        )
        loss = loss.view(bsz, -1).sum(1)
        loss += sel_loss

        if reduce:
            loss = loss.sum()
            nll_loss = nll_loss.sum()
            sel_loss = sel_loss.sum()
        return loss, nll_loss, sel_loss, sel_entropy, batch_entropy

    def update_best_selection(self, model, net_outputs, sample):
        lprobs = model.get_normalized_probs(net_outputs[0], log_probs=True)
        bsz = lprobs.size(0)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # We compute the target loglikelihoods via nll_loss function
        target = model.get_targets(sample, net_outputs[0]).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx,
            reduce=False,
        )
        loglikelihoods = -nll_loss.view(bsz, -1, 1).sum(1)

        sel_lprobs = [compute_sel_lprobs(out) for out in net_outputs]    

        # we reduce the selections to a single list of matrices (~layers)
        # so we can extract best selection for each matrix (~layer)
        n_selections = len(net_outputs[0][1]['selections'])
        selections = []
        for i in range(n_selections):
            sel = None
            if net_outputs[0][1]['selections'][i] is not None:
                sel = torch.stack([
                    out[1]['selections'][i]
                    for out in net_outputs], axis=1)
            selections.append(sel)

        joint_lprobs = loglikelihoods + torch.stack(sel_lprobs, axis=-1)
        best_sel_indices = joint_lprobs.max(1)[1].detach()
        best_sel_indices = [list(range(bsz)), best_sel_indices]
        model.update_best_selection(
            [sel[best_sel_indices] if sel is not None else None
             for sel in selections])

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ctrl_loss_sum = sum(log.get('ctrl_loss', 0) for log in logging_outputs)
        sel_entropy = sum(log.get('sel_entropy', 0) for log in logging_outputs)
        batch_entropy = sum(log.get('batch_entropy', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('ctrl_loss', ctrl_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('sel_entropy', sel_entropy, 1, round=3)
        metrics.log_scalar('batch_entropy', batch_entropy, 1, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
