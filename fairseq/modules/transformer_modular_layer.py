# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from torch import Tensor

from fairseq import utils
from fairseq.modules import (
    MaskedFeedForwardBlock,
    MaskedMultiheadAttention,
)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.modular import ModularCtrl
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase,
    TransformerEncoderLayerBase,
)


class TransformerModularEncoderLayer(TransformerEncoderLayerBase):
    """
    Encoder layer block with conditional computation support.

    The base block is extended with additional controller subnetworks which
    handle the masking of the mask-supporting submodules (e.g. masked multi-head
    attention) based on its input.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    def __init__(self, cfg, return_fc=False):
        super().__init__(cfg, return_fc)
        self.ctrl_ffn = None
        if cfg.module_ctrl.encoder_ffn:
            assert cfg.encoder.ffn_modules is not None
            self.ctrl_ffn = self.build_ffn_controller(self.embed_dim, cfg)

        self.ctrl_self = None
        if cfg.module_ctrl.encoder_attn:
            self.ctrl_self = self.build_self_attn_controller(self.embed_dim, cfg)

    def build_ffn(self, embed_dim, cfg):
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )

        return MaskedFeedForwardBlock(
            embed_dim,
            cfg.encoder.ffn_embed_dim,
            activation_fn=self.activation_fn,
            num_modules=cfg.encoder.ffn_modules,
            dropout_module=self.dropout_module,
            activation_dropout_module=activation_dropout_module,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_ffn_controller(self, embed_dim, cfg):
        return ModularCtrl(
            embed_dim,
            cfg.encoder.ffn_modules,
            self.activation_fn,
            hidden_depth=cfg.module_ctrl.hidden_depth,
            hidden_dim=cfg.module_ctrl.hidden_dim,
            dropout=cfg.dropout,
            word_dropout=cfg.module_ctrl.word_dropout,
            use_hard_samples=cfg.module_ctrl.use_hard_samples,
            add_output_bias=cfg.module_ctrl.add_output_bias,
            input_average_pooling=cfg.module_ctrl.input_average_pooling,
        )

    def build_self_attention(self, embed_dim, cfg):
        return MaskedMultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def build_self_attn_controller(self, embed_dim, cfg):
        return ModularCtrl(
            embed_dim,
            cfg.encoder.attention_heads,
            self.activation_fn,
            hidden_depth=cfg.module_ctrl.hidden_depth,
            hidden_dim=cfg.module_ctrl.hidden_dim,
            dropout=cfg.dropout,
            word_dropout=cfg.module_ctrl.word_dropout,
            use_hard_samples=cfg.module_ctrl.use_hard_samples,
            add_output_bias=cfg.module_ctrl.add_output_bias,
            input_average_pooling=cfg.module_ctrl.input_average_pooling,
        )

    def forward(
        self,
        x,
        encoder_padding_mask,
        attn_mask: Optional[Tensor] = None,
        module_mask: Optional[Tensor] = None,
        ctrl_temperature: Tensor = 1.,
        need_head_weights: bool = False,
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
            module_mask: a fixed controller output module mask, whenever
                a controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            ctrl_temperature: temperature parameter for Gumbel-Softmax
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
        Returns:
            tuple of:
                encoded output of shape `(seq_len, batch, embed_dim)`
                controller module outputs
        """
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        ctrl_self_out = None
        self_mod_mask = module_mask
        if self_mod_mask is not None:
            self_mod_mask = self_mod_mask.view(1, 1, -1)
        elif self.ctrl_self is not None:
            ctrl_self_out = self.ctrl_self(
                x,
                encoder_padding_mask,
                ctrl_temperature
            )
            self_mod_mask = ctrl_self_out["mask"]

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            module_mask=self_mod_mask,
            need_weights=False,
            need_head_weights=need_head_weights,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
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
                ctrl_temperature
            )
            ffn_mod_mask = ctrl_ffn_out["mask"]

        x, fc_result = self.ffn(x, module_mask=ffn_mod_mask)
        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.return_fc and not torch.jit.is_scripting():
            return (
                x,
                fc_result,
                {"encoder_attn" : ctrl_self_out, "encoder_ffn": ctrl_ffn_out}
            )

        return (
            x, {"encoder_attn" : ctrl_self_out, "encoder_ffn": ctrl_ffn_out}
        )


class TransformerModularDecoderLayer(TransformerDecoderLayerBase):
    """
    Decoder layer block with conditional computation support.

    The base block is extended with additional controller subnetworks which
    handle the masking of the mask-supporting submodules (e.g. masked multi-head
    attention) based on its input.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.ctrl_ffn = None
        if cfg.module_ctrl.decoder_ffn:
            assert cfg.decoder.ffn_modules is not None
            self.ctrl_ffn = self.build_ffn_controller(self.embed_dim, cfg)

        self.ctrl_self = None
        if cfg.module_ctrl.decoder_attn:
            self.ctrl_self = self.build_self_attn_controller(self.embed_dim, cfg)

        self.ctrl_enc = None
        if cfg.module_ctrl.encdec_attn:
            self.ctrl_enc = self.build_encoder_attn_controller(self.embed_dim, cfg)

    def build_ffn(self, embed_dim, cfg):
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )

        return MaskedFeedForwardBlock(
            embed_dim,
            cfg.decoder.ffn_embed_dim,
            activation_fn=self.activation_fn,
            num_modules=cfg.decoder.ffn_modules,
            dropout_module=self.dropout_module,
            activation_dropout_module=activation_dropout_module,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_ffn_controller(self, embed_dim, cfg):
        return ModularCtrl(
            embed_dim,
            cfg.decoder.ffn_modules,
            self.activation_fn,
            hidden_depth=cfg.module_ctrl.hidden_depth,
            hidden_dim=cfg.module_ctrl.hidden_dim,
            dropout=cfg.dropout,
            word_dropout=cfg.module_ctrl.word_dropout,
            use_hard_samples=cfg.module_ctrl.use_hard_samples,
            add_output_bias=cfg.module_ctrl.add_output_bias,
            input_average_pooling=cfg.module_ctrl.input_average_pooling,
        )

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MaskedMultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def build_self_attn_controller(self, embed_dim, cfg):
        return ModularCtrl(
            embed_dim,
            cfg.decoder.attention_heads,
            self.activation_fn,
            hidden_depth=cfg.module_ctrl.hidden_depth,
            hidden_dim=cfg.module_ctrl.hidden_dim,
            dropout=cfg.dropout,
            word_dropout=cfg.module_ctrl.word_dropout,
            use_hard_samples=cfg.module_ctrl.use_hard_samples,
            add_output_bias=cfg.module_ctrl.add_output_bias,
            input_average_pooling=cfg.module_ctrl.input_average_pooling,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MaskedMultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attn_controller(self, embed_dim, cfg):
        return ModularCtrl(
            embed_dim,
            cfg.decoder.attention_heads,
            self.activation_fn,
            hidden_depth=cfg.module_ctrl.hidden_depth,
            hidden_dim=cfg.module_ctrl.hidden_dim,
            dropout=cfg.dropout,
            word_dropout=cfg.module_ctrl.word_dropout,
            use_hard_samples=cfg.module_ctrl.use_hard_samples,
            add_output_bias=cfg.module_ctrl.add_output_bias,
            input_average_pooling=cfg.module_ctrl.input_average_pooling
        )

    def forward(
        self,
        x,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[Tensor]] = None,
        prev_attn_state: Optional[List[Tensor]] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        module_mask: Optional[Tensor] = None,
        ctrl_temperature: Tensor = 1.,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            module_mask: a fixed controller output module mask, whenever
                an controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            ctrl_temperature: temperature parameter for Gumbel-Softmax
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            tuple of:
                encoded output of shape `(seq_len, batch, embed_dim)`
                controller module outputs
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
                temperature=ctrl_temperature
            )
            self_mod_mask = ctrl_self_out["mask"]

        x, _ = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
            module_mask=self_mod_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
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
                    temperature=ctrl_temperature
                )
                enc_mod_mask = ctrl_enc_out["mask"]

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                module_mask=enc_mod_mask,
            )
            x =self.dropout_module(x)
            x = self.residual_connection(x, residual)
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
                temperature=ctrl_temperature
            )
            ffn_mod_mask = ctrl_ffn_out["mask"]

        x, _ = self.ffn(x, module_mask=ffn_mod_mask)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
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
                x, attn, self_attn_state,
                {
                    "decoder_attn": ctrl_self_out,
                    "encdec_attn": ctrl_enc_out,
                    "decoder_ffn": ctrl_ffn_out
                }
            )
        return (
            x, attn, None,
            {
                "decoder_attn": ctrl_self_out,
                "encdec_attn": ctrl_enc_out,
                "decoder_ffn": ctrl_ffn_out
            }
        )
