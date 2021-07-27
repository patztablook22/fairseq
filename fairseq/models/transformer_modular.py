# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, NamedTuple, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from fairseq import options, utils
from fairseq.models import (
    register_model,
    register_model_architecture,
    transformer
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    ModularCtrlOut,
    SinusoidalPositionalEmbedding,
    TransformerModularEncoderLayer,
    TransformerModularDecoderLayer
)
from torch import Tensor


logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer_modular")
class TransformerModularModel(TransformerModel):
    """
    Transformer architecture with an additional conditional computation support.

    See the parent class for more details.
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        super(TransformerModularModel, TransformerModularModel).add_args(parser)
        parser.add_argument('--module-ctrl-type', type=str,
                            help='type of TransformerModular layer to be used')
        parser.add_argument('--module-ctrl-avg-tokens', action='store_true',
                            help='average controler input sequence')
        parser.add_argument('--module-ctrl-hard-samples', action='store_true',
                            help='use hard (0, 1) modular mask samples during training')
        parser.add_argument('--module-ctrl-hidden-depth', type=int,
                            help='num of controller hidden layers')
        parser.add_argument('--module-ctrl-hidden-dim', type=int,
                            help='controller hidden dimension size')
        parser.add_argument('--module-ctrl-word-dropout', type=float,
                            help='controller word dropout')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        transformer_modular(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerModularEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerModularDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        module_mask: Optional[Dict[str, Tensor]] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        ctrl_temperature: Tensor = 1.,
    ):
        """
        Args:
            module_mask: a fixed controller output module mask, whenever
                an controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            ctrl_temperature: temperature parameter for Gumbel-Softmax
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            module_mask=module_mask,
            return_all_hiddens=return_all_hiddens,
            ctrl_temperature=ctrl_temperature,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            module_mask=module_mask,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            ctrl_temperature=ctrl_temperature
        )
        extra['attn_weights']['encoder'] = encoder_out.enc_self_attn_weights
        for key, value in encoder_out.ctrl_outputs.items():
            extra['ctrl_outputs'][key] = value

        return x, extra


EncoderModularOut = NamedTuple(
    "EncoderModularOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("enc_self_attn_weights", Optional[List[Tensor]]),  # List[T x T]
        ("ctrl_outputs", Dict[str, List[Optional[Tensor]]]),  # ModularCtrlOut
    ]
)


class TransformerModularEncoder(TransformerEncoder):
    """Transformer encoder with an additional conditional computation support."""

    def build_encoder_layer(self, args):
        return TransformerModularEncoderLayer(args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        module_mask: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        ctrl_temperature: Tensor = 1.,
    ):
        """
        Args:
            module_mask: a fixed controller output module mask, whenever
                an controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            ctrl_temperature: temperature parameter for Gumbel-Softmax
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None
        attn_weights = []
        enc_ctrl_outputs = []

        # encoder layers
        for layer in self.layers:
            x, attn_w, ctrl_out = layer(
                x, module_mask, encoder_padding_mask,
                need_head_weights=True, ctrl_temperature=ctrl_temperature)
            attn_weights.append(attn_w)
            enc_ctrl_outputs.append(ctrl_out)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderModularOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            enc_self_attn_weights=attn_weights,
            src_tokens=None,
            src_lengths=None,
            ctrl_outputs={ 'encoder': enc_ctrl_outputs },
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderModularOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        ctrl_outputs = encoder_out.ctrl_outputs
        new_ctrl_outputs = {}
        for key in ctrl_outputs:
            new_ctrl_outputs[key] = []
            for ctrl_out in ctrl_outputs[key]:
                if ctrl_out is None:
                    new_ctrl_outputs[key].append(ctrl_out)
                else:
                    new_logits = ctrl_out.logits.index_select(0, new_order)
                    new_mask = ctrl_out.mask.index_select(0, new_order)
                    new_ctrl_out = ModularCtrlOut(
                        logits=new_logits,
                        mask=new_mask
                    )
                    new_ctrl_outputs[key].append(new_ctrl_out)

        # TODO: Should we also reorder attn_weights?
        return EncoderModularOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            enc_self_attn_weights=encoder_out.enc_self_attn_weights,
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
            ctrl_outputs=new_ctrl_outputs,
        )


class TransformerModularDecoder(TransformerDecoder):
    """Transformer decoder with an additional conditional computation support."""

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerModularDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        module_mask: Optional[Dict[str, Tensor]] = None,
        return_all_hiddens: bool = False,
        ctrl_temperature: Tensor = 1.,
    ):
        """
        Args:
            module_mask: a fixed controller output module mask, whenever
                an controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            ctrl_temperature: temperature parameter for Gumbel-Softmax
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            module_mask=module_mask,
            ctrl_temperature=ctrl_temperature,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        module_mask: Optional[Dict[str, Tensor]] = None,
        ctrl_temperature: Tensor = 1.,
    ):
        """
        Args:
            module_mask: a fixed controller output module mask, whenever
                an controller is defined in a Transformer layer, a controller
                output will be ignored and the module_mask will be used instead
            ctrl_temperature: temperature parameter for Gumbel-Softmax
        """

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        self_attn_weights = []
        enc_attn_weights = []

        self_ctrl_outputs = []
        enc_ctrl_outputs = []

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, self_attn, enc_attn, _, ctrl_out = layer(
                x,
                module_mask,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                ctrl_temperature=ctrl_temperature,
            )

            self_attn_weights.append(self_attn)
            enc_attn_weights.append(enc_attn)

            self_ctrl_outputs.append(ctrl_out['decoder'])
            enc_ctrl_outputs.append(ctrl_out['enc_dec'])

            inner_states.append(x)
            if enc_attn is not None and idx == alignment_layer:
                attn = enc_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        attn_weights = {
            "decoder": self_attn_weights,
            "enc_dec": enc_attn_weights,
        }
        return x, {
            "attn": [attn],
            "attn_weights": attn_weights,
            "inner_states": inner_states,
            "ctrl_outputs": {
                "decoder": self_ctrl_outputs,
                "enc_dec": enc_ctrl_outputs,
            },
        }


@register_model_architecture("transformer_modular", "transformer_modular")
def transformer_modular(args):
    args.module_ctrl_type = getattr(args, 'module_ctrl_type', 'attention')
    args.module_ctrl_avg_tokens = getattr(args, 'module_ctrl_avg_tokens', False)
    args.module_ctrl_hard_samples = getattr(args, 'module_ctrl_hard_samples', False)

    args.module_ctrl_hidden_depth = getattr(
        args, 'module_ctrl_hidden_depth', 0)
    args.module_ctrl_hidden_dim = getattr(
        args, 'module_ctrl_hidden_dim', None)
    args.module_ctrl_word_dropout = getattr(
        args, 'module_ctrl_word_dropout', 0.0)

    transformer.base_architecture(args)
