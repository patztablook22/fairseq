# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, NamedTuple, Optional, Tuple
import logging
import itertools

import torch
import torch.nn.init as init
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import Parameter
from fairseq.distributions import FactoredCategorical
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.quant_noise import quant_noise


logger = logging.getLogger(__name__)


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

    def forward_alt(self, x, selection, time_major=True):
        # Alternative version of the forward method.
        # Performance-wise, this version seems inferior to the current forward method.
        assert selection.dim() == 2

        if time_major:
            x = x.transpose(0, 1)

        selection_unique = torch.unique(selection, dim=0)
        outputs = None
        for i in range(selection_unique.size(0)):
            sel = selection_unique[i]
            assert sel.max() < self.n_modules

            mask = selection.eq(sel).min(-1).values.float()
            mask = mask.view(-1, 1, 1).to(x.device)

            w = self.weight[sel]
            if self.sum_outputs:
                w = torch.cat([k for k in w], axis=-1)
            else:
                w = w.view(-1, self.weight.size(-1))

            b = self.bias
            if b is not None:
                b = self.bias[sel]
                if self.sum_outputs:
                    b = b.sum(0)
                b = b.view(-1)

            o = F.linear(x, w, b)
            o *= mask
            if outputs is None:
                outputs = o
            else:
                outputs += o

        if time_major:
            outputs = outputs.transpose(0, 1)

        return outputs

    def forward(self, x, selection, time_major=True):
        assert selection.dim() == 2

        if time_major:
            x = x.transpose(0, 1)

        selection_unique = torch.unique(selection)
        if self.sum_outputs:
            x = x.view(x.size(0), x.size(1), selection.size(1), self.in_features)

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
                 dim,
                 n_modules,
                 n_active,
                 ctrl_type='joint'):
        super().__init__()
        self.dim = dim
        self.n_modules = n_modules
        self.n_active = n_active
        self.ctrl_type = ctrl_type

        self.subsets = None
        if ctrl_type == 'joint':
            self.subsets = list(itertools.combinations(range(n_modules), n_active))
            self.subsets = torch.tensor(self.subsets, dtype=torch.long)
            self.out_proj = nn.Linear(dim, self.subsets.size(0))
        elif ctrl_type == 'factored':
            self.out_proj = nn.Linear(dim, n_modules * n_active)
        else:
            raise ValueError('Invalid module controller type ({})'.format(ctrl_type))

        self.best_prediction = None
        self.reset_parameters()

    def forward(self, x, mode, padding_mask=None, indices=None):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1, self.dim)
        if padding_mask is not None:
            # The mask contains '1' in place of the padding symbols
            mask = (~padding_mask).long()
            mask_flat = mask.view(batch_size, -1, 1)
            x_flat *= mask_flat
        x_flat = x_flat.sum(1)

        logits = self.out_proj(x_flat)
        #if torch.isnan(x).any():
        #    logger.warn('Controller input vector contains NaN')
        if self.ctrl_type == 'factored':
            logits = logits.view(-1, self.n_active, self.n_modules)
        elif self.ctrl_type == 'joint':
            logits = logits.unsqueeze(-2)
        ctrl = Categorical(logits=logits)

        # batch_size x 1 x subset_size
        if mode == 'e_step':
            ctrl_prediction = ctrl.sample([1])[0]
        elif mode == 'm_step':
            assert indices is not None
            ctrl_prediction = self.get_best_prediction(indices)
        elif mode == 'validation':
            ctrl_prediction = logits.max(-1)[1]
        else:
            raise ValueError('Invalid ModuleCtrl mode: {}'.format(mode))

        ctrl_prediction = ctrl_prediction.to(x.device)
        selection = self.pred2sel(ctrl_prediction)

        return ModularCtrlOut(
            ctrl=ctrl,
            selection=selection,
            ctrl_prediction=ctrl_prediction,
        )

    def pred2sel(self, prediction):
        if self.subsets is not None:
            return self.subsets[prediction].squeeze(1)
        return prediction

    def initialize_best_prediction(self, dataset_size):
        if self.subsets is not None:
            sel_size = self.subsets.size(0)
            self.best_prediction = torch.LongTensor(
                dataset_size, 1).random_(0, sel_size)
        else:
            self.best_prediction = torch.LongTensor(
                dataset_size, self.n_active).random_(0, self.n_modules)

    def update_best_prediction(self, sel, indices):
        self.best_prediction[indices] = sel.to(self.best_prediction.device)

    def get_best_prediction(self, indices):
        assert self.best_prediction is not None
        return self.best_prediction[indices]

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.out_proj.weight, gain=1 / math.sqrt(2))

class ModularMultiheadAttention(MultiheadAttention):
    """TODO"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_active,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self_attention,
            encoder_decoder_attention=encoder_decoder_attention,
            q_noise=q_noise,
            qn_block_size=qn_block_size)
        assert num_heads_active <= self.num_heads
        self.num_heads_active = num_heads_active

        if q_noise > 0.:
            logger.warning('quant_noise modules are not properly supported (tested) at the moment')
        self.k_proj = quant_noise(
            ModularLinear(self.kdim, self.head_dim, self.num_heads, bias=bias),
            q_noise, qn_block_size)
        self.v_proj = quant_noise(
            ModularLinear(self.vdim, self.head_dim, self.num_heads, bias=bias),
            q_noise, qn_block_size)
        self.q_proj = quant_noise(
            ModularLinear(embed_dim, self.head_dim, self.num_heads, bias=bias),
            q_noise, qn_block_size)
           
        self.out_proj = quant_noise(
            ModularLinear(
                self.head_dim, embed_dim, self.num_heads, bias=bias, sum_outputs=True),
            q_noise, qn_block_size)

        self.enable_torch_version = False
        #TODO: support add_bias_kv
        #if add_bias_kv:
        #    self.bias_k = Parameter(
        #        torch.Tensor(n_modules, 1, 1, embed_dim))
        #    self.bias_v = Parameter(
        #        torch.Tensor(n_modules, 1, 1, embed_dim))
        #else:
        self.bias_k = self.bias_v = None

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        selection: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        TODO: mode/indices description
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            self.enable_torch_version
            and not self.onnx_trace
            and incremental_state is None
            and not static_kv
        ):
            raise ValueError('``enable_torch_version'' option is not supported')

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query, selection)
            k = self.k_proj(query, selection)
            v = self.v_proj(query, selection)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query, selection)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key, selection)
                v = self.v_proj(key, selection)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query, selection)
            k = self.k_proj(key, selection)
            v = self.v_proj(value, selection)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            #TODO extract bias_k, bias_v + concat
            #k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            #v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads_active, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads_active, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads_active, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads_active, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads_active, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = ModularMultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads_active, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads_active, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = ModularMultiheadAttention.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads_active, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads_active, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf")
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads_active, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, selection, ctrl

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads_active, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.head_dim * self.num_heads_active)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.head_dim * self.num_heads_active)
        attn = self.out_proj(attn, selection)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads_active, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights
