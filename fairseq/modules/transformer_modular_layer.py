# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import (
    ModularMultiheadAttention,
    TransformerEncoderLayer
)
from torch import Tensor


class TransformerModularEncoderLayer(TransformerEncoderLayer):
    """TODO

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    # TODO: build fc1, fc2 with modular options

    def build_self_attention(self, embed_dim, args):
        ctrl_type = None
        if args.module_ctrl_type == 'joint-separate':
            ctrl_type = 'joint'
        elif args.module_ctrl_type == 'factored-separate':
            ctrl_type = 'factored'

        return ModularMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            args.encoder_attention_heads_active,
            dropout=args.attention_dropout,
            self_attention=True,
            ctrl_type=ctrl_type,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size)

    def initialize_best_selection(self, dataset_size):
        self.self_attn.initialize_best_selection(dataset_size)

    def update_best_selection(self, sel, indices):
        self.self_attn.update_best_selection(sel, indices)

    def get_best_selection(self, indices):
        return self.self_attn.get_best_selection(indices)

    def forward(
        self,
        x,
        encoder_padding_mask,
        mode: str,
        sel_indices = None,
        indices: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            mode: TODO
            sel_indices: TODO
            indices: TODO
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attentioni
            need_head_weights: TODO

        Returns:
            tuple of:
                encoded output of shape `(seq_len, batch, embed_dim)`
                selected attention heads
                controller module
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x, attn_weights, selection, ctrl = self.self_attn(
            query=x, key=x, value=x, mode=mode, sel_indices=sel_indices,
            indices=indices, attn_mask=attn_mask,
            key_padding_mask=encoder_padding_mask,
            need_head_weights=need_head_weights
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights, selection, ctrl
