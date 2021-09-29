import math
from typing import Dict, NamedTuple, Optional
import logging
import itertools
import random
import numpy as np

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import Parameter

from fairseq.incremental_decoding_utils import with_incremental_state


_EPS = 1e-9


def masked_mean(x, mask, axis=None, keepdim=False):
    x *= mask.float()
    return x.sum(axis, keepdim=keepdim) / (mask.sum(axis, keepdim=keepdim) + _EPS)


def sample_gumbel(shape, device='cpu'):
    U_1 = torch.rand(shape).to(device)
    U_2 = torch.rand(shape).to(device)
    return -torch.log(
        (torch.log(U_1 + _EPS) - _EPS) / (torch.log(U_2 + _EPS) - _EPS)
    )


def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return torch.sigmoid(y / temperature)


def gaussian_sigmoid_sample(logits, temperature):
    # We want to progressively increase the noise during the training
    # to force more hard outputs (the temperature is always decreasing)
    y = logits + (torch.normal(0., 1., logits.size()) / temperature)
    return torch.sigmoid(y)


def gumbel_sigmoid(logits, temperature=1.0):
    return gumbel_sigmoid_sample(logits, temperature)


def gaussian_sigmoid(logits, temperature=1.0):
    return gaussian_sigmoid(logits, temperature)


ModularCtrlOut = NamedTuple(
    "ModularCtrlOut",
    [
        ("logits", Tensor),
        ("sampled_probs", Tensor),
        ("mask", Tensor),
    ]
)


@with_incremental_state
class ModularCtrl(nn.Module):
    """
    Module controller for conditional computation.

    This module takes an input sequence before it is passed to a masked
    Transformer module (e.g. masked multi-head attention) and produces
    a module mask for the said module.

    This controller is an implementation of deep averaging network (DAN) described in
    Iyyer et al. (2015), "Deep unordered composition rivals syntactic methods for text classification."

    Args:
        n_modules: number of modules/heads
        hidden_depth: number of hidden layers
        hidden_dim: size of the hidden layers
        word_dropout: a probability of dropping the whole input representation
            from the input sequence (see Iyyer et al., 2015)
        hard_samples: apply hard thresholding (0 or 1) on the Gumbel-Softmax output
        averaged_tokens: average the input sequence and produce a single mask
            for the whole sequence (in decoder, in a given timestep)
    """
    def __init__(self,
                 input_dim,
                 n_modules,
                 hidden_depth=0,
                 hidden_dim=None,
                 dropout=0.0,
                 word_dropout=0.0,
                 hard_samples=False,
                 averaged_tokens=False,
                 add_output_bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.hard_samples = hard_samples
        self.averaged_tokens = averaged_tokens
        self.add_output_bias = add_output_bias

        self.n_modules = n_modules

        modules = []
        for _ in range(hidden_depth):
            if hidden_dim is None:
                raise ValueError("controller hidden_dim cannot be NoneType if hidden_depth > 0")
            modules.append(nn.Linear(input_dim, hidden_dim))
            # TODO: change to GeLU?
            # TODO: pass via parameter
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        self.fc_net = nn.Sequential(*modules)

        self.dropout = dropout
        self.word_dropout = word_dropout
        self.out_proj = nn.Linear(input_dim, n_modules, bias=self.add_output_bias)

    def extract_features(self, x, padding_mask=None, future_mask=None):
        """Compute the pre-softmax representation."""
        # shape(x) = (bsz, seq_len, input_dim)
        x_len = x.size(1)

        if padding_mask is not None:
            # The mask contains '1' in place of the padding symbols
            input_mask = ~padding_mask
            input_mask = input_mask.unsqueeze(-1)
        else:
            input_mask = torch.ones(x.size(0), x.size(1), 1, device=x.device)
        input_mask = input_mask.float()

        # Word dropout described in Iyyer et al. (2015)
        if self.training:
            input_mask *= torch.bernoulli(
                torch.ones(input_mask.shape, device=x.device) * (1. - self.word_dropout))

        if self.averaged_tokens:
            if future_mask is not None:
                future_mask = future_mask.reshape(1, x_len, x_len, 1)
                x *= input_mask.float()
                x = x.unsqueeze(1).repeat([1, x_len, 1, 1])
                x = masked_mean(x, mask=future_mask, axis=2)
            else:
                x = masked_mean(x, mask=input_mask, axis=1, keepdim=True)
        else:
            x *= input_mask.float()

        return self.fc_net(x)

    def forward(
        self,
        x,
        padding_mask: Optional[Tensor] = None,
        future_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        temperature=1.0
    ) -> ModularCtrlOut:
        """
        Compute a module mask based on the input sequence x.

        Args:
            padding_mask: sequence padding mask
            future_mask: mask for the following tokens
            temperature: temperature for Gumbel-Softmax

        Returns:
            tuple of:
                logits: mask logits of `(batch, seq_len, n_modules)` or `(batch, 1, n_modules)`
                sampled_probs: probability of the positive mask value based from logits (+ sampled noise during
                    training), same shape as logits
                module_mask: hard-thresholded sampled_probs (same shape as logits)
        """
        x = x.transpose(0, 1)

        # HACK: we want to guarantee that the temperature will always be positive
        # TODO: find out which scheduler can set the non-positive value
        if temperature <= 0:
            temperature = _EPS

        if future_mask is not None:
            # future_mask contains values to mask decoder self-attn "before_softmax"
            # we need to set 1 to nonmasked tokens (0. in the original)
            # and 0 to masked ones (-inf in the original) instead
            future_mask = (future_mask == 0.).float()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        if saved_state is not None:
            # (prev_x) saved states are stored with shape (bsz, seq_len, input_dim)
            if "prev_x" in saved_state:
                prev_x = saved_state["prev_x"]
                assert prev_x is not None
                x = torch.cat([prev_x, x], dim=1)

            prev_padding_mask: Optional[Tensor] = None
            if "prev_padding_mask" in saved_state:
                prev_padding_mask = saved_state["prev_padding_mask"]
            assert x is not None
            padding_mask = ModularCtrl._append_prev_padding_mask(
                padding_mask=padding_mask,
                prev_padding_mask=prev_padding_mask,
                batch_size=x.size(0),
                seq_len=x.size(1),
            )

            saved_state["prev_x"] = x
            saved_state["prev_padding_mask"] = padding_mask

            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        features = self.extract_features(x, padding_mask, future_mask)
        logits = self.out_proj(features)

        if saved_state is not None:
            logits = logits[:, -x.size(1):]

        if self.training:
            sampled_probs = gumbel_sigmoid(logits, temperature)
            sampled_probs = ModularCtrl._mask_output_probs(sampled_probs, padding_mask)
            module_mask = sampled_probs
            if self.hard_samples:
                module_mask = (module_mask > 0.5).float()
                module_mask = (module_mask - sampled_probs).detach() + sampled_probs

        else:
            sampled_probs = torch.sigmoid(logits)
            sampled_probs = ModularCtrl._mask_output_probs(sampled_probs, padding_mask)
            module_mask = (sampled_probs > threshold).float()

        return ModularCtrlOut(
            logits=logits,
            sampled_probs=sampled_probs,
            mask=module_mask
        )

    @staticmethod
    def _append_prev_padding_mask(
        padding_mask: Optional[Tensor],
        prev_padding_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_padding_mask is not None and padding_mask is not None:
            new_padding_mask = torch.cat(
                [prev_padding_mask.float(), padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, seq_len - prev_padding_mask.size(1)),
                device=prev_padding_mask.device,
            )
            new_padding_mask = torch.cat(
                [prev_padding_mask.float(), filler.float()], dim=1
            )
        elif padding_mask is not None:
            filler = torch.zeros(
                (batch_size, seq_len - padding_mask.size(1)),
                device=padding_mask.device,
            )
            new_padding_mask = torch.cat(
                [filler.float(), padding_mask.float()], dim=1
            )
        else:
            new_padding_mask = prev_padding_mask
        return new_padding_mask

    @staticmethod
    def _mask_output_probs(x, padding_mask):
        if padding_mask is None:
            return x
        padding_mask = ~padding_mask.unsqueeze(-1)
        if x.size(1) != padding_mask.size(1):
            return x * torch.any(padding_mask.bool(), 1, keepdim=True).float()
        return x * padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "ctrl_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "ctrl_state", buffer)
