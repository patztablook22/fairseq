# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn, distributions
from torch.nn import Parameter
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.incremental_decoding_utils import with_incremental_state


def _tile(x, dim, n_tile):
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    return x.repeat(repeat_idx)


class ModularLinear(nn.Module):
    """TODO"""

    def __init__(self,
                 in_features,
                 out_features,
                 n_modules,
                 bias=False,
                 concat_dim=1):
        super().__init__()
        self.n_modules = n_modules
        self.concat_dim = concat_dim
        assert concat_dim in [0, 1]

        self.weight = Parameter(
            torch.Tensor(n_modules, in_features, out_features))
        self.bias = None
        if bias:
            self.bias =  Parameter(
                torch.Tensor(n_modules, out_features))

    def forward(self, x, selection, time_major=True):
        assert selection.dim() == 2

        if time_major:
            x = x.transpose(0, 1)

        outputs = []
        for i in range(selection.size(0)):
            sel = selection[i]
            assert sel.max() < self.n_modules

            w = torch.cat([x for x in self.weight[sel]], self.concat_dim)
            o = torch.matmul(x[i], w)
            if self.bias is not None:
                if self.concat_dim == 0:
                    o += self.bias[sel].sum(0).view(1, -1)
                else:
                    o += self.bias[sel].view(1, -1)
            outputs.append(o)

        outputs = torch.stack(outputs, dim=0)
        if time_major:
            outputs = outputs.transpose(0, 1)

        return outputs


class ModularCtrlV2(nn.Module):
    """TODO"""

    def __init__(self,
                 dim,
                 n_modules,
                 n_active,
                 sample_size=1,
                 allow_repetition=True):
        super().__init__()
        self.dim = dim
        self.n_modules = n_modules
        self.n_active = n_active
        self.sample_size = sample_size
        self.allow_repetition = allow_repetition

        self.ctrl_proj = nn.Linear(dim, n_modules * n_active)

        self.best_sel = None

    def forward(self, x, mode, indices=None):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1, self.dim).sum(1)

        logits = self.ctrl_proj(x_flat)
        logits = logits.view(-1, self.n_active, self.n_modules)
        ctrl = distributions.categorical.Categorical(logits=logits)

        # batch_size x sample_size x subset_size
        if mode == 'e_step':
            selection = ctrl.sample([self.sample_size])[0]
        elif mode == 'm_step':
            assert indices is not None
            selection = self.get_best_selection(indices)
        elif mode == 'validation':
            selection = logits.max(-1)[1].view(-1, self.n_active)
        else:
            raise ValueError('({}) Invalid mode'.format(self.__name__))
        selection = selection.to(x.device)

        return selection, ctrl

    def initialize_best_selection(self, dataset_size):
        self.best_sel = torch.LongTensor(
            dataset_size, self.n_active).random_(0, self.n_modules)

    def update_best_selection(self, sel, indices):
        self.best_sel[indices] = sel.to(self.best_sel.device)

    def get_best_selection(self, indices):
        assert self.best_sel is not None
        return self.best_sel[indices]


class ModularMultiheadAttentionV2(MultiheadAttention):
    """TODO"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        n_modules,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        sample_size=1
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False)

        assert self.num_heads <= n_modules

        self.k_proj = ModularLinear(
            self.kdim, self.head_dim, n_modules, bias=bias)
        self.v_proj = ModularLinear(
            self.vdim, self.head_dim, n_modules, bias=bias)
        self.q_proj = ModularLinear(
            embed_dim, self.head_dim, n_modules, bias=bias)

        self.out_proj = ModularLinear(
            self.head_dim, embed_dim, n_modules, bias=bias, concat_dim=0)

        self.module_ctrl = ModularCtrlV2(
            embed_dim, n_modules, num_heads, sample_size)

        self.enable_torch_version = False
        #TODO: support add_bias_kv
        #if add_bias_kv:
        #    self.bias_k = Parameter(
        #        torch.Tensor(n_modules, 1, 1, embed_dim))
        #    self.bias_v = Parameter(
        #        torch.Tensor(n_modules, 1, 1, embed_dim))
        #else:
        self.bias_k = self.bias_v = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def initialize_best_selection(self, dataset_size):
        self.module_ctrl.initialize_best_selection(dataset_size)

    def update_best_selection(self, sel, indices):
        self.module_ctrl.update_best_selection(sel, indices)

    def get_best_selection(self, indices):
        return self.module_ctrl.get_best_selection(indices)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        mode: str,
        indices: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        selection, ctrl = self.module_ctrl(query.transpose(0, 1), mode, indices)

        if (
            self.enable_torch_version
            and not self.onnx_trace
            and incremental_state is None
            and not static_kv
        ):
            raise ValueError("``enable_torch_version'' option is not supported")

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
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            #TODO saved_state support
            raise ValueError('``saved_state'' option not supported')
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = ModularMultiheadAttentionV2._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
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
        attn_weights = ModularMultiheadAttentionV2.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

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
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn, selection)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, selection, ctrl

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:

            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1))
            if key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order
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
        result = self.get_incremental_state(incremental_state, "attn_state")
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
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
