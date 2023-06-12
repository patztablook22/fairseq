# Feedforward Blocks used in Transformer FFN layer impelementations
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class FeedForwardBlock(nn.Module):
    """
    Wrapper for the feedforward Transformer block.

    TODO
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        activation_fn: Callable,
        dropout_module: Optional[FairseqDropout] = None,
        activation_dropout_module: Optional[FairseqDropout] = None,
        ffn_layernorm: Optional[LayerNorm] = None,
        bias: bool = True,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.activation_fn = activation_fn
        self.dropout_module = dropout_module
        self.activation_dropout_module = activation_dropout_module
        self.ffn_layernorm = ffn_layernorm
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.fc1 = self.build_fc1(embed_dim, ffn_embed_dim, bias)
        self.fc2 = self.build_fc2(ffn_embed_dim, embed_dim, bias)

    def build_fc1(self, input_dim: int, output_dim: int, bias: bool):
        return quant_noise(
            nn.Linear(input_dim, output_dim, bias=bias),
            p=self.q_noise, block_size=self.qn_block_size
        )

    def build_fc2(self, input_dim: int, output_dim: int, bias: bool):
        return quant_noise(
            nn.Linear(input_dim, output_dim, bias=bias),
            p=self.q_noise, block_size=self.qn_block_size
        )

    def _get_fc_rank(self, remove_num: int) -> List[int]:
        f1_filter_param = []
        for i in range(self.fc1.out_features):
            f1_filter_param.append(
                torch.sum(torch.abs(self.fc1.weight[i]))
                + torch.sum(torch.abs(self.fc2.weight[:, i]))
                + torch.abs(self.fc1.bias[i])
            )
        return sorted(
            range(len(f1_filter_param)), key=lambda k: f1_filter_param[k], reverse=False
        )[0:remove_num]

    def _prune_fc_layer(self, remove_index: List[int]):
        new_fc1_weight = []
        new_fc1_bias = []
        for i in range(self.fc1.out_features):
            if i not in remove_index:
                new_fc1_weight.append(self.fc1.weight[i])
                new_fc1_bias.append(self.fc1.bias[i])

        new_fc1_weight = torch.stack(new_fc1_weight).detach()
        new_fc1_weight.requires_grad = True

        new_fc1_bias = torch.stack(new_fc1_bias).detach()
        new_fc1_bias.requires_grad = True

        self.fc1 = quant_noise(
            nn.Linear(self.fc1.in_features, self.fc1.out_features - len(remove_index)),
            p=self.q_noise,
            block_size=self.qn_block_size,
        )
        self.fc1.weight = torch.nn.Parameter(new_fc1_weight)
        self.fc1.bias = torch.nn.Parameter(new_fc1_bias)

        new_fc2_weight = []
        new_fc2_bias = []
        for i in range(self.fc2.in_features):
            if i not in remove_index:
                new_fc2_weight.append(self.fc2.weight[:, i])
        new_fc2_bias = self.fc2.bias.detach()

        new_fc2_weight = torch.stack(new_fc2_weight, dim=-1).detach()
        new_fc2_weight.requires_grad = True

        new_fc2_bias = self.fc2.bias.detach()
        new_fc2_bias.requires_grad = True

        self.fc2 = quant_noise(
            nn.Linear(self.fc2.in_features - len(remove_index), self.fc2.out_features),
            p=self.q_noise,
            block_size=self.qn_block_size,
        )
        self.fc2.weight = torch.nn.Parameter(new_fc2_weight)
        self.fc2.bias = torch.nn.Parameter(new_fc2_bias)

    def forward(self, x: Tensor) -> Tensor:
        """TODO"""
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)

        fc_result = x

        x = self.dropout_module(x)
        return x, fc_result


class MaskedFeedForwardBlock(FeedForwardBlock):
    """
    Wrapper for the feedforward Transformer block allowing module masking.

    TODO
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        activation_fn: Callable,
        num_modules: int = 1,
        dropout_module: Optional[FairseqDropout] = None,
        activation_dropout_module: Optional[FairseqDropout] = None,
        ffn_layernorm: Optional[LayerNorm] = None,
        bias: bool = True,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ):
        super().__init__(
            embed_dim,
            ffn_embed_dim,
            activation_fn,
            dropout_module,
            activation_dropout_module,
            ffn_layernorm,
            bias,
            q_noise,
            qn_block_size
        )
        self.num_modules = num_modules
        self.module_dim = ffn_embed_dim // num_modules
        assert (
            self.module_dim * self.num_modules == self.ffn_embed_dim
        ), "ffn_embed_dim must be divisible by num_modules"

    def forward(self, x: Tensor, module_mask: Tensor) -> Tensor:
        """TODO"""
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        if module_mask is not None:
            x_len, bsz, ffn_embed_dim = x.size()
            x = x.contiguous().view(x_len, bsz, self.num_modules, self.module_dim)

            module_mask = module_mask.transpose(0, 1).unsqueeze(-1)
            x = x * module_mask
            x = x.view(x_len, bsz, self.ffn_embed_dim)
        x = self.fc2(x)

        fc_result = x

        x = self.dropout_module(x)
        return x, fc_result
