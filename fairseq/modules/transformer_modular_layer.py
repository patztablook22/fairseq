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
    FeedForwardBlock,
    ModularCtrl,
    ModularCtrlOut,
    MaskedMultiheadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from torch import Tensor


class TransformerModularEncoderLayer(TransformerEncoderLayer):
    """
    Encoder layer block with conditional computation support.

    The base block is extended with additional controller subnetworks which
    handle the masking of the mask-supporting submodules (e.g. masked multi-head
    attention) based on its input.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    def __init__(self, args):
        super().__init__(args)
        self.ctrl_ffn = None
        if args.encoder_ffn_modules is not None:
            self.ctrl_ffn = self.build_ffn_controller(args)

        self.ctrl_self = None
        if args.module_ctrl_encoder_attn:
            self.ctrl_self = self.build_self_attn_controller(args)

    def build_ffn(self, embed_dim, args):
        activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )

        activation_dropout = getattr(args, "activation_dropout", 0.0)
        if activation_dropout == 0.0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout = getattr(args, "relu_dropout", 0.0)

        return MaskedFeedForwardBlock(
            embed_dim,
            args.encoder_ffn_embed_dim,
            args.encoder_ffn_modules,
            activation_fn=activation_fn,
            dropout=self.dropout,
            activation_dropout=activation_dropout,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size)

    def build_ffn_controller(self, args):
        return ModularCtrl(
            args.encoder_embed_dim,
            args.encoder_ffn_modules,
            hidden_depth=args.module_ctrl_hidden_depth,
            hidden_dim=args.module_ctrl_hidden_dim,
            dropout=args.dropout,
            word_dropout=args.module_ctrl_word_dropout,
            hard_samples=args.module_ctrl_hard_samples,
            add_output_bias=args.module_ctrl_add_output_bias,
            averaged_tokens=args.module_ctrl_avg_tokens)

    def build_self_attention(self, embed_dim, args):
        return MaskedMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_self_attn_controller(self, args):
        return ModularCtrl(
            args.encoder_embed_dim,
            args.encoder_attention_heads,
            hidden_depth=args.module_ctrl_hidden_depth,
            hidden_dim=args.module_ctrl_hidden_dim,
            dropout=args.dropout,
            word_dropout=args.module_ctrl_word_dropout,
            hard_samples=args.module_ctrl_hard_samples,
            add_output_bias=args.module_ctrl_add_output_bias,
            averaged_tokens=args.module_ctrl_avg_tokens)

    def forward(
        self,
        x,
        module_mask: Tensor,
        encoder_padding_mask,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
        ctrl_temperature: Tensor = 1.,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            module_mask: a fixed controller output module mask, whenever
                a controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attentioni
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
            ctrl_temperature: temperature parameter for Gumbel-Softmax

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

        ctrl_self_out = None
        self_mod_mask = module_mask
        if self_mod_mask is not None:
            self_mod_mask = self_mod_mask.view(1, 1, -1)
        elif self.ctrl_self is not None:
            ctrl_self_out = self.ctrl_self(
                x,
                encoder_padding_mask,
                ctrl_temperature)
            self_mod_mask = ctrl_self_out.mask

        x, attn_weights = self.self_attn(
            query=x, key=x, value=x, module_mask=self_mod_mask,
            attn_mask=attn_mask,
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

        ctrl_ffn_out = None
        ffn_mod_mask = module_mask
        if ffn_mod_mask is not None:
            ffn_mod_mask = ffn_mod_mask.view(1, 1, -1)
        elif self.ctrl_ffn is not None:
            ctrl_ffn_out = self.ctrl_ffn(
                x,
                encoder_padding_mask,
                ctrl_temperature)
            ffn_mod_mask = ctrl_ffn_out.mask

        x = self.ffn(x, module_mask=ffn_mod_mask)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x, attn_weights,
            {"encoder_attn" : ctrl_self_out, "encoder_ffn": ctrl_ffn_out}
        )


class TransformerModularDecoderLayer(TransformerDecoderLayer):
    """
    Decoder layer block with conditional computation support.

    The base block is extended with additional controller subnetworks which
    handle the masking of the mask-supporting submodules (e.g. masked multi-head
    attention) based on its input.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.ctrl_ffn = None
        if args.decoder_ffn_modules is not None:
            self.ctrl_ffn = self.build_ffn_controller(args)

        self.ctrl_self = None
        if args.module_ctrl_decoder_attn:
            self.ctrl_self = self.build_self_attn_controller(args)

        self.ctrl_enc = None
        if args.module_ctrl_encdec_attn:
            self.ctrl_enc = self.build_encoder_attn_controller(args)

    def build_ffn(self, embed_dim, args):
        activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )

        activation_dropout = getattr(args, "activation_dropout", 0.0)
        if activation_dropout == 0.0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout = getattr(args, "relu_dropout", 0.0)

        return MaskedFeedForwardBlock(
            embed_dim,
            args.decoder_ffn_embed_dim,
            args.decoder_ffn_modules,
            activation_fn=activation_fn,
            dropout=self.dropout,
            activation_dropout=activation_dropout,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size)

    def build_ffn_controller(self, args):
        return ModularCtrl(
            args.decoder_embed_dim,
            args.decoder_ffn_modules,
            hidden_depth=args.module_ctrl_hidden_depth,
            hidden_dim=args.module_ctrl_hidden_dim,
            dropout=args.dropout,
            word_dropout=args.module_ctrl_word_dropout,
            hard_samples=args.module_ctrl_hard_samples,
            add_output_bias=args.module_ctrl_add_output_bias,
            averaged_tokens=args.module_ctrl_avg_tokens)

    def build_self_attention(self,
                             embed_dim,
                             args,
                             add_bias_kv=False,
                             add_zero_attn=False):
        return MaskedMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MaskedMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_self_attn_controller(self, args):
        return ModularCtrl(
            args.decoder_embed_dim,
            args.decoder_attention_heads,
            hidden_depth=args.module_ctrl_hidden_depth,
            hidden_dim=args.module_ctrl_hidden_dim,
            dropout=args.dropout,
            word_dropout=args.module_ctrl_word_dropout,
            hard_samples=args.module_ctrl_hard_samples,
            add_output_bias=args.module_ctrl_add_output_bias,
            averaged_tokens=args.module_ctrl_avg_tokens)

    def build_encoder_attn_controller(self, args):
        return ModularCtrl(
            args.decoder_embed_dim,
            args.decoder_attention_heads,
            hidden_depth=args.module_ctrl_hidden_depth,
            hidden_dim=args.module_ctrl_hidden_dim,
            dropout=args.dropout,
            word_dropout=args.module_ctrl_word_dropout,
            hard_samples=args.module_ctrl_hard_samples,
            add_output_bias=args.module_ctrl_add_output_bias,
            averaged_tokens=args.module_ctrl_avg_tokens)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        module_mask: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        ctrl_temperature: Tensor = 1.,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            module_mask: a fixed controller output module mask, whenever
                an controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
            ctrl_temperature: temperature parameter for Gumbel-Softmax

        Returns:
            tuple of:
                encoded output of shape `(seq_len, batch, embed_dim)`
                selected attention heads
                controller module output
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        ctrl_self_out = None
        self_mod_mask = module_mask
        if self_mod_mask is not None:
            self_mod_mask = self_mod_mask.view(1, 1, -1)
        elif self.ctrl_self is not None:
            ctrl_self_out = self.ctrl_self(
                x,
                padding_mask=self_attn_padding_mask,
                future_mask=self_attn_mask,
                incremental_state=incremental_state,
                temperature=ctrl_temperature)
            self_mod_mask = ctrl_self_out.mask

        x, attn_weights_self = self.self_attn(
            query=x,
            key=y,
            value=y,
            module_mask=self_mod_mask,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            ctrl_enc_out = None
            enc_mod_mask = module_mask
            if enc_mod_mask is not None:
                enc_mod_mask = enc_mod_mask.view(1, 1, -1)
            elif self.ctrl_enc is not None:
                ctrl_enc_out = self.ctrl_enc(
                    x,
                    padding_mask=self_attn_padding_mask,
                    future_mask=self_attn_mask,
                    incremental_state=incremental_state,
                    temperature=ctrl_temperature)
                enc_mod_mask = ctrl_enc_out.mask

            x, attn_weights_enc = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                module_mask=enc_mod_mask,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        ctrl_ffn_out = None
        ffn_mod_mask = module_mask
        if ffn_mod_mask is not None:
            ffn_mod_mask = ffn_mod_mask.view(1, 1, -1)
        elif self.ctrl_ffn is not None:
            ctrl_ffn_out = self.ctrl_ffn(
                x,
                padding_mask=self_attn_padding_mask,
                future_mask=self_attn_mask,
                incremental_state=incremental_state,
                temperature=ctrl_temperature)
            ffn_mod_mask = ctrl_ffn_out.mask

        x = self.ffn(x, module_mask=ffn_mod_mask)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return (
                x, attn_weights_self, attn_weights_enc, self_attn_state,
                {
                    "decoder_attn": ctrl_self_out,
                    "enc_dec_attn": ctrl_enc_out,
                    "decoder_ffn": ctrl_ffn_out
                }
            )
        return (
            x, attn_weights_self, attn_weights_enc, None,
            {
                "decoder_attn": ctrl_self_out,
                "enc_dec_attn": ctrl_enc_out,
                "decoder_ffn": ctrl_ffn_out
            }
        )

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Scriptable reorder incremental state in transformer layers."""
        self.self_attn.reorder_incremental_state(incremental_state, new_order)

        if self.encoder_attn is not None:
            self.encoder_attn.reorder_incremental_state(incremental_state, new_order)

        if self.ctrl_self is not None:
            self.ctrl_self.reorder_incremental_state(incremental_state, new_order)

        if self.ctrl_enc is not None:
            self.ctrl_enc.reorder_incremental_state(incremental_state, new_order)


class MaskedFeedForwardBlock(FeedForwardBlock):
    """
    Wrapper for the feedforward Transformer block allowing module masking.

    TODO
    """
    def __init__(self,
                 embed_dim,
                 ffn_embed_dim,
                 num_modules=None,
                 activation_fn=F.relu,
                 dropout=0.0,
                 activation_dropout=0.0,
                 bias=True,
                 q_noise=0.0,
                 qn_block_size=8):
        super().__init__(
            embed_dim,
            ffn_embed_dim,
            activation_fn,
            dropout,
            activation_dropout,
            bias,
            q_noise,
            qn_block_size)
        # HACK: without FFN controller (num_modules is None) we fall back
        # to ``regular'' FFN block
        self.num_modules = 1
        if num_modules is not None:
            self.num_modules = num_modules

        self.module_dim = ffn_embed_dim // self.num_modules
        assert (
            self.module_dim * self.num_modules == self.ffn_embed_dim
        ), "ffn_embed_dim must be divisible by num_modules"

    def forward(self, x, module_mask: Tensor) -> Tensor:
        """TODO"""
        x_proj = self.activation_fn(self.fc1(x))
        x_proj = F.dropout(x_proj, p=float(self.activation_dropout), training=self.training)

        if module_mask is not None:
            x_len, bsz, ffn_embed_dim = x_proj.size()
            x_proj = x_proj.contiguous().view(x_len, bsz, self.num_modules, self.module_dim)

            module_mask = module_mask.transpose(0, 1).unsqueeze(-1)
            x_proj = x_proj * module_mask
            x_proj = x_proj.view(x_len, bsz, self.ffn_embed_dim)

        y = self.fc2(x_proj)
        return F.dropout(y, p=self.dropout, training=self.training)
