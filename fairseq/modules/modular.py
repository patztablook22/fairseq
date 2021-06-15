import math
from typing import NamedTuple, Optional
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


_EPS = 1e-6


def masked_mean(x, mask, axis=None):
    x *= mask.float()
    return x.sum(axis) / (mask.sum(axis) + _EPS)


def sample_gumbel(shape, device='cpu'):
    U_1 = torch.rand(shape).to(device)
    U_2 = torch.rand(shape).to(device)
    return -torch.log(torch.log(U_1) / torch.log(U_2))


def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return torch.sigmoid(y / temperature)


def gumbel_sigmoid(logits, temperature=1.0):
    y = gumbel_sigmoid_sample(logits, temperature)
    y_hard = (y > 0.5).float()
    return (y_hard - y).detach() + y


ModularCtrlOut = NamedTuple(
    "ModularCtrlOut",
    [
        ("logits", Tensor),
        ("mask", Tensor),
    ]
)


class ModularCtrl(nn.Module):
    """TODO"""

    def __init__(self,
                 input_dim,
                 n_modules,
                 hidden_depth=0,
                 hidden_dim=None,
                 dropout=0.0,
                 word_dropout=0.0,
                 averaged_tokens=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.averaged_tokens = averaged_tokens

        self.n_modules = n_modules

        modules = []
        for _ in range(hidden_depth):
            if hidden_dim is None:
                raise ValueError("controller hidden_dim cannot be NoneType if hidden_depth > 0")
            modules.append(nn.Linear(input_dim, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        self.fc_net = nn.Sequential(*modules)

        self.dropout = dropout
        self.word_dropout = word_dropout
        self.out_proj = nn.Linear(input_dim, n_modules)

    def extract_features(self, x, padding_mask=None):
        bsz = x.size(0)
        x = x.view(bsz, -1, self.input_dim)

        if padding_mask is not None:
            # The mask contains '1' in place of the padding symbols
            mask = ~padding_mask
            mask = mask.view(bsz, -1, 1)
        else:
            mask = torch.ones(x.size(0), x.size(1), 1).to(x.device)
        mask = mask.float()

        # Word dropout described in Iyyer et al. (2015)
        if self.training:
            mask *= torch.bernoulli(
                torch.ones(mask.shape).to(x.device) * (1. - self.word_dropout))

        if self.averaged_tokens:
            x = masked_mean(x, mask=mask, axis=0).unsqueeze(0)
        else:
            x *= mask.float()

        return self.fc_net(x)

    def forward(self, x, padding_mask=None):
        bsz = x.size(0)
        features = self.extract_features(x, padding_mask)
        logits = self.out_proj(features)

        if self.training:
            mask = gumbel_sigmoid(logits)
        else:
            mask = (logits > 0.).float()

        return ModularCtrlOut(
            logits=logits,
            mask=mask
        )
