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
logger = logging.getLogger(__name__)


def masked_mean(x, mask, axis=None):
    x *= mask.float()
    return x.sum(axis) / (mask.sum(axis) + _EPS)


def initialize_proportional(n, indices):
    assert len(indices) > 0
    if len(indices) == 1:
        return [indices[0]] * n
    n /= 2
    return [indices[0]] * math.ceil(n) + initialize_proportional(int(n), indices[1:])


class ModularLinear(nn.Module):
    """TODO"""

    __constants__ = ['in_features', 'out_features', 'n_modules']

    def __init__(self,
                 in_features,
                 out_features,
                 n_modules,
                 bias=False,
                 sum_outputs=False):
        super(ModularLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_modules = n_modules
        self.sum_outputs = sum_outputs

        self.weight = Parameter(torch.Tensor(
            n_modules, out_features, in_features))
        if bias:
            if sum_outputs:
                # Output projection has a single shared bias
                self.bias = Parameter(torch.Tensor(1, out_features))
            else:
                self.bias = Parameter(torch.Tensor(n_modules, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(
                    self.weight)
                bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, selection, time_major=True):
        assert selection.dim() == 2

        if time_major:
            x = x.transpose(0, 1)

        selection_unique = torch.unique(selection)
        if self.sum_outputs:
            x = x.view(
                x.size(0), x.size(1), selection.size(1), self.in_features)

            # TODO: can we do this with a single matrix multiplication?
            outputs = None
            for i in selection_unique:
                mask = selection.eq(i).to(x.device)
                mask = mask.unsqueeze(1).unsqueeze(-1)

                o = F.linear(x, self.weight[i], None)
                o *= mask

                if outputs is None:
                    outputs = o
                else:
                    outputs += o
            outputs = outputs.sum(2)
            if self.bias is not None:
                outputs += self.bias
        else:
            w = self.weight.view(-1, self.weight.size(-1))
            b = None
            if self.bias is not None:
                b = self.bias.view(-1)
            o = F.linear(x, w, b)

            o = o.view(o.size(0), o.size(1), self.n_modules, self.out_features)
            o = o.transpose(1, 2)
            outputs = [
                o[[[k for k in range(selection.size(0))], selection[:, i]]]
                for i in range(selection.size(1))]
            outputs = torch.cat(outputs, -1)

        if time_major:
            outputs = outputs.transpose(0, 1)

        return outputs

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, n_modules={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.n_modules
        )


ModularCtrlOut = NamedTuple(
    "ModularCtrlOut",
    [
        ("ctrl", Tensor),  # Categorical
        ("selection", Tensor),  # B x n_active
        ("ctrl_prediction", Optional[Tensor]),
    ]
)


class ModularCtrl(nn.Module):
    """TODO"""

    def __init__(self,
                 input_dim,
                 n_modules,
                 n_active,
                 hidden_depth=0,
                 hidden_dim=None,
                 dropout=0.0,
                 word_dropout=0.0,
                 ctrl_type='joint'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth

        self.n_modules = n_modules
        self.n_active = n_active
        self.ctrl_type = ctrl_type

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

        self.subsets = None
        if ctrl_type == 'joint':
            self.subsets = list(itertools.combinations(range(n_modules), n_active))
            self.subsets = torch.tensor(self.subsets, dtype=torch.long)
            self.out_proj = nn.Linear(input_dim, self.subsets.size(0))
        elif ctrl_type == 'factored':
            self.out_proj = nn.Linear(input_dim, n_modules * n_active)
        else:
            raise ValueError("Invalid module controller type ({})".format(ctrl_type))

        self.best_selection = None
        self.reset_parameters()

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

        x = masked_mean(x, mask=mask, axis=1)

        return self.fc_net(x)

    def forward(self, x, mode, padding_mask=None, indices=None, fixed_selection=None):
        bsz = x.size(0)
        features = self.extract_features(x, padding_mask)
        logits = self.out_proj(features)

        if self.ctrl_type == 'factored':
            logits = logits.view(-1, self.n_active, self.n_modules)
        elif self.ctrl_type == 'joint':
            logits = logits.unsqueeze(-2)
        ctrl = Categorical(logits=logits)

        # batch_size x 1 x subset_size
        if mode == 'e_step':
            ctrl_prediction = ctrl.sample([1])[0].to(x.device)
            selection = self.pred2sel(ctrl_prediction)
        elif mode == 'm_step':
            assert indices is not None
            selection = self.get_best_selection(indices).to(x.device)
            ctrl_prediction = self.sel2pred(selection)
        elif mode == 'validation':
            ctrl_prediction = logits.max(-1)[1].to(x.device)
            selection = self.pred2sel(ctrl_prediction)
        elif mode == 'full':
            ctrl = None
            ctrl_prediction = None
            selection = torch.arange(self.n_modules).repeat(bsz, 1)
            ctrl_prediction = self.sel2pred(selection)
        else:
            raise ValueError("Invalid ModuleCtrl mode: {}".format(mode))

        if fixed_selection is not None:
            selection = fixed_selection
            if mode != 'validation' and fixed_selection.size(-1) != self.n_active:
                raise ValueError(
                    "Size of ``fixed_selection'' and ``n_active'' must be "
                    "equal outside of the ``validation'' controller mode")
            elif mode != 'validation':
                ctrl_prediction =  self.sel2pred(selection)

        selection = selection.to(x.device)
        ctrl_prediction = ctrl_prediction.to(x.device)
        return ModularCtrlOut(
            ctrl=ctrl,
            selection=selection,
            ctrl_prediction=ctrl_prediction,
        )

    def list_all_predictions(self):
        if self.subsets is not None:
            return torch.arange(0, len(self.subsets)).long()
        raise ValueError(
            "list_all_predictions not supported for ctrl_type=='factored'")

    def list_all_selections(self):
        if self.subsets is not None:
            return self.subsets
        raise ValueError(
            "list_all_selections not supported for ctrl_type=='factored'")

    def pred2sel(self, prediction):
        if self.subsets is not None:
            return self.subsets[prediction].squeeze(1)
        return prediction

    def sel2pred(self, selection):
        if self.subsets is not None:
            if self.subsets.size(1) != selection.size(1):
                # TODO: This is quite ugly workaround
                return torch.tensor([-1]).repeat(self.subsets.size(0))
            return torch.cat(
                [
                    torch.nonzero((s.to(self.subsets.device) == self.subsets).prod(1), as_tuple=False)
                    for s in selection
                ], 0)
        return selection

    def initialize_best_selection(self, dataset_size, init_scheme='uniform'):
        if self.subsets is not None:
            sel_size = self.subsets.size(0)
            if init_scheme == 'uniform':
                prediction = torch.LongTensor(dataset_size, 1).random_(0, sel_size)
                self.best_selection = self.pred2sel(prediction)
            elif init_scheme == 'proportional':
                indices = list(range(sel_size))
                random.shuffle(indices)
                prediction = initialize_proportional(dataset_size, indices)
                random.shuffle(prediction)
                prediction = torch.tensor(prediction).unsqueeze(1)
                assert prediction.size(0) == dataset_size
                self.best_selection = self.pred2sel(prediction)
            else:
                # You can initialize with a fixed selection, e.g. '(head_1,head_2)'
                try:
                    selection = eval(init_scheme)
                    if type(selection) is int:
                        selection = [selection]
                    self.best_selection = torch.tensor(
                        selection).unsqueeze(0).repeat(dataset_size, 1)
                except:
                    raise ValueError(
                        "Unknown ModularCtrl initialization scheme "
                        "'{}'".format(init_scheme))
        else:
            self.best_selection = torch.LongTensor(
                dataset_size, self.n_active).random_(0, self.n_modules)

    def update_best_selection(self, sel, indices):
        self.best_selection[indices] = sel.to(self.best_selection.device)

    def get_best_selection(self, indices):
        assert self.best_selection is not None
        return self.best_selection[indices]

    def get_best_selection_stats(self):
        res = {}
        for sel in self.best_selection:
            sel = ",".join(sel.cpu().numpy().astype(np.str))
            if sel not in res:
                res[sel] = 1
            else:
                res[sel] += 1
        return res

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.out_proj.weight, gain=1 / math.sqrt(2))
