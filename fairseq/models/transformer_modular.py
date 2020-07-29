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
from fairseq import options, utils
from fairseq.models import (
    register_model,
    register_model_architecture,
    transformer
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder
)
from fairseq.modules import (
    ModularCtrl,
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer,
    TransformerModularEncoderLayer
)
from torch import Tensor


logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer_modular")
class TransformerModularModel(TransformerModel):
    """
    TODO

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Modular Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        super(TransformerModularModel, TransformerModularModel).add_args(parser)
        parser.add_argument('--encoder-attention-heads-active', type=int,
                            metavar='N',
                            help='size of the attention head pool')
        parser.add_argument('--encoder-modular-layer-indices',
                            help='tuple of indices of modular layers')
        parser.add_argument('--module-ctrl-type', type=str,
                            help='type of the module controller')
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

    def initialize_best_selection(self, dataset_size):
        self.encoder.initialize_best_selection(dataset_size)

    def update_best_selection(self, selections, indices):
        self.encoder.update_best_selection(selections, indices)

    def get_best_selection(self, indices):
        return self.get_best_selection(indices)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        mode: str = None,
        indices: Optional[Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        TODO: mode/indices description
        """
        if mode is None:
            logger.warning('mode==None in forward method, using mode=validation as default')
            mode = 'validation'

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            mode=mode,
            indices=indices,
            return_all_hiddens=return_all_hiddens,
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        extra["enc_self_attn_weights"] = encoder_out.enc_self_attn_weights
        extra["selections"] = encoder_out.selections
        extra["controllers"] = encoder_out.controllers
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
        ("selections", List[Optional[Tensor]]),  # List[B x H]
        ("controllers", List[Optional[Any]]),  # List[Categorical]
    ]
)


class TransformerModularEncoder(TransformerEncoder):
    """
    TODO

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.module_ctrl = None
        if args.module_ctrl_type == 'joint-shared':
            self.module_ctrl = ModularCtrl(
                args.encoder_embed_dim, args.encoder_attention_heads,
                args.encoder_attention_heads_active, 'joint')
        elif args.module_ctrl_type == 'factored-shared':
            self.module_ctrl = ModularCtrl(
                args.encoder_embed_dim, args.encoder_attention_heads,
                args.encoder_attention_heads_active, 'factored')

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([
            self.build_encoder_layer(args, i)
            for i in range(args.encoder_layers)
        ])
        self.num_layers = len(self.layers)

    def build_encoder_layer(self, args, i=None):
        modular_layer_indices = eval(args.encoder_modular_layer_indices)
        if type(modular_layer_indices) is int:
            modular_layer_indices = [modular_layer_indices]

        if i in modular_layer_indices:
            return TransformerModularEncoderLayer(args)
        else:
            return TransformerEncoderLayer(args)

    def forward_layer_wrapper(self,
                              layer,
                              x,
                              encoder_padding_mask,
                              mode,
                              sel_indices,
                              indices,
                              need_head_weights):
        if not isinstance(layer, TransformerModularEncoderLayer):
            res = layer(x, encoder_padding_mask, need_head_weights=need_head_weights)
            return res[0], res[1], None, None
        else:
            return layer(x, encoder_padding_mask, mode, sel_indices, indices, need_head_weights=need_head_weights)

    def initialize_best_selection(self, dataset_size):
        if self.module_ctrl is not None:
            self.module_ctrl.initialize_best_selection(dataset_size)
        else:
            for i, _ in enumerate(self.layers):
                if isinstance(
                    self.layers[i], TransformerModularEncoderLayer
                ):
                    self.layers[i].initialize_best_selection(dataset_size)

    def update_best_selection(self, selections, indices):
        if self.module_ctrl is not None:
            self.module_ctrl.update_best_selection(selections[0], indices)
        else:
            assert len(selections) == len(self.layers)
            for i, sel in enumerate(selections):
                if sel is not None:
                    self.layers[i].update_best_selection(sel, indices)

    def get_best_selection(self, indices):
        if self.module_ctrl is not None:
            sel = self.module_ctrl.get_best_selection(indices)
            return [sel for _ in self.layers]
        else:
            selections = []
            for i, layer in enumerate(self.layers):
                if isinstance(
                    self.layers[i], TransformerModularEncoderLayer
                ):
                    selections.append(self.layers[i].get_best_selection(indices))
                else:
                    selections.append(None)
            return selections

    def forward(
        self,
        src_tokens,
        src_lengths,
        mode: str = None,
        indices: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        TODO: mode/indices description
        """
        if mode is None:
            logger.warning('mode==None in forward method, using mode=validation as default')
            mode = 'validation'

        x, encoder_embedding = self.forward_embedding(src_tokens)

        sel_indices = None
        attn_weights = []
        selections = []
        controllers = []
        if self.module_ctrl is not None:
            selection, ctrl = self.module_ctrl(x, mode, indices)
            sel_indices = self.module_ctrl.sel2indices(selection)
            selections = [selection]
            controllers = [ctrl]

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x, attn_w, selection, ctrl = self.forward_layer_wrapper(
                layer, x, encoder_padding_mask, mode, sel_indices, indices, need_head_weights=True)
            attn_weights.append(attn_w)
            if self.module_ctrl is None:
                selections.append(selection)
                controllers.append(ctrl)
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
            src_tokens=None,
            src_lengths=None,
            enc_self_attn_weights=attn_weights,
            selections=selections,  # List[B x H]
            controllers=controllers,  # List[]
        )


@register_model_architecture("transformer_modular", "transformer_modular")
def transformer_modular(args):
    args.encoder_attention_heads_active = getattr(
        args, 'encoder_attention_heads_active', args.encoder_attention_heads)
    args.module_ctrl_type = getattr(args, 'module_ctrl_type', 'joint-shared')
    args.encoder_modular_layer_indices = getattr(
        args, 'encoder_modular_layer_indices', '()')

    transformer.base_architecture(args)
