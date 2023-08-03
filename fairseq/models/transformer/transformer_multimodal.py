# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
from torch import nn, Tensor

import logging

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model
from fairseq.models.transformer import (
    TransformerMultimodalConfig,
    TransformerModelBase,
    TransformerDecoderBase,
    TransformerEncoderBase,
)
from fairseq.modules import (
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    transformer_layer,
)


logger = logging.getLogger(__name__)


@register_model("transformer_multimodal", dataclass=TransformerMultimodalConfig)
class TransformerMultimodalModel(TransformerModelBase):
    """
    Transformer architecture with an additional conditional computation support.

    See the parent class for more details.

    Args:
        encoder (TransformerMultimodalEncoder): the encoder
        decoder (TransformerMultimodalDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerMultimodalConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(
            cfg, src_dict, encoder_embed_tokens, task
        )
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens, task):
        return TransformerMultimodalEncoder(cfg, src_dict, embed_tokens, task)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerMultimodalDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        src_images,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_images,
            src_lengths=src_lengths,
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
        return x, extra


class TransformerMultimodalEncoder(TransformerEncoderBase):
    """
    Transformer encoder with an additional conditional computation support.

    See the parent class for more details.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """
    def __init__(self, cfg, dictionary, embed_tokens, task, return_fc=False):
        self.cfg = cfg
        super().__init__(cfg, dictionary, embed_tokens, return_fc=return_fc)

        self.patch_image_size = task.patch_image_size
        self.patch_size = cfg.encoder.patch_size

        if self.patch_image_size % self.patch_size != 0:
            raise ValueError(
                "the resized image size ({}) has to be divisible by the image "
                "encoder patch_size ({})".format(image_size, patch_size)
            )

        self.embed_images = nn.Conv2d(
            in_channels=3,
            out_channels=embed_tokens.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.embed_image_positions = (
            PositionalEmbedding(
                (self.patch_image_size // self.patch_size) ** 2,
                embed_tokens.embedding_dim,
                padding_idx=None,
                learned=True,
            )
        )

        self.input_image_text_concat = cfg.encoder.input_image_text_concat
        if self.input_image_text_concat and not cfg.encoder.shared_text_image_encoder:
            raise ValueError(
                "Requires ``model.encoder.shared_text_image_encoder=true'' "
                "if model.encoder.input_image_text_concat is enabled."
            )

        if cfg.encoder.shared_text_image_encoder:
            self.vit_layers = self.layers
            self.vit_layer_norm = self.layer_norm
        else:
            if self.encoder_layerdrop > 0.0:
                self.vit_layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.vit_layers = nn.ModuleList([])
            self.vit_layers.extend(
                [sel.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
            )
            self.num_layers = len(self.layers)
            if cfg.encoder.normalize_before:
                self.vit_layer_norm = LayerNorm(embed_tokens.embedding_dim, export=cfg.export)
            else:
                self.vit_layer_norm = None

    def forward_embedding(
        self, src_tokens, src_images, token_embedding: Optional[Tensor] = None
    ):
        # embed tokens/images
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
            image_embedding = self.embed_images(src_images)

            # reshape the image embedding
            image_embedding = image_embedding.reshape(
                image_embedding.shape[0], image_embedding.shape[1], -1
            ).permute(0, 2, 1)

        # scale
        x = embed = self.embed_scale * token_embedding
        x_img = self.embed_scale * image_embedding

        # + position info
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        # TODO: other than linear positions?
        img_positions = (
            torch.arange(
                x_img.shape[1]
            ).unsqueeze(0).expand(
                x_img.shape[0], x_img.shape[1]
            ).to(x_img.device)
        )
        x_img += self.embed_image_positions(
            x_img,
            positions=img_positions,
        )

        # layer norm
        # TODO: user separate LN for tokens and images?
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
            x_img = self.layernorm_embedding(x_img)

        # dropout
        x = self.dropout_module(x)
        x_img = self.dropout_module(x_img)

        # quantization
        if self.quant_noise is not None:
            x = self.quant_noise(x)
            x_img = self.quant_noise(x_img)
        return x, x_img, embed

    def forward(
        self,
        src_tokens,
        src_images,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_images (FloatTensor): TODO
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens,
            src_images,
            src_lengths,
            return_all_hiddens,
            token_embeddings,
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_images,
        src_lengths: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_images (FloatTensor): TODO
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (
            torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        )
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        x, x_img, encoder_embedding = self.forward_embedding(
            src_tokens, src_images, token_embeddings
        )
        encoder_image_padding_mask = torch.zeros(
            [x_img.shape[0], x_img.shape[1]]
        ).to(x_img.device)

        # account for padding while computing the representation
        x = x * (
            1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x_img = x_img.transpose(0, 1)

        if self.input_image_text_concat:
            x = torch.cat([x_img, x], axis=0)
            encoder_padding_mask = torch.cat([encoder_image_padding_mask, encoder_padding_mask], axis=1)
        else:
            # TODO: non-concatenated version
            raise NotImplementedError(
                "Seperate processing of image and text is currently not "
                "implemented. Use ``cfg.encoder.input_image_text_concat=True'' "
                "instead."
            )

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer, vit_layer in zip(self.layers, self.vit_layers):
            lr = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                need_head_weights=False,
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_images": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        if len(encoder_out["src_images"]) == 0:
            src_images = []
        else:
            src_images = [(encoder_out["src_images"]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_images": src_images,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class TransformerMultimodalDecoder(TransformerDecoderBase):
    """
    Transformer decoder with an additional conditional computation support.

    See the parent class for more details.

    Args:
        cfg (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    pass
