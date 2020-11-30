#!/usr/bin/env python3

import argparse
import torch

from fairseq import utils
from fairseq.file_io import PathManager
from fairseq import checkpoint_utils
from fairseq.modules.modular_multihead_attention import ModularCtrl


def convert_model(model, ns, coder="encoder", att_type="self_attn"):
        embed_dim = getattr(ns, "{}_embed_dim".format(coder))
        modular_layers = eval(getattr(ns, "{}_modular_layer_indices".format(coder)))
        n_heads = getattr(ns, "{}_attention_heads".format(coder))
        n_layers = getattr(ns, "{}_layers".format(coder))

        for i in modular_layers:
            p_template = "{}.layers.{}." + att_type + ".in_proj_weight"

            if p_template.format(coder, i) in model:
                p_template = "{}.layers.{}." + att_type + ".in_proj_{}"

                q_weight, k_weight, v_weight = model[
                    p_template.format(coder, i, "weight")].chunk(3, 0)
                q_bias, k_bias, v_bias = model[
                    p_template.format(coder, i, "bias")].chunk(3, 0)

                del model[p_template.format(coder, i, "weight")]
                del model[p_template.format(coder, i, "bias")]
            else:
                p_template = "{}.layers.{}." + att_type + ".{}_proj.{}"

                q_weight = model[p_template.format(coder, i, "q", "weight")]
                k_weight = model[p_template.format(coder, i, "k", "weight")]
                v_weight = model[p_template.format(coder, i, "v", "weight")]

                q_bias = model[p_template.format(coder, i, "q", "bias")]
                k_bias = model[p_template.format(coder, i, "k", "bias")]
                v_bias = model[p_template.format(coder, i, "v", "bias")]

            p_template = "{}.layers.{}." + att_type + ".{}_proj.{}"
            
            out_weight = model[p_template.format(coder, i, "out", "weight")]
            out_bias = model[p_template.format(coder, i, "out", "bias")]

            # Reshape the parameters
            model[p_template.format(coder, i, "q", "weight")] = q_weight.view(n_heads, -1, embed_dim)
            model[p_template.format(coder, i, "k", "weight")] = k_weight.view(n_heads, -1, embed_dim)
            model[p_template.format(coder, i, "v", "weight")] = v_weight.view(n_heads, -1, embed_dim)

            model[p_template.format(coder, i, "q", "bias")] = q_bias.view(n_heads, -1)
            model[p_template.format(coder, i, "k", "bias")] = k_bias.view(n_heads, -1)
            model[p_template.format(coder, i, "v", "bias")] = v_bias.view(n_heads, -1)

            out_weight = out_weight.transpose(0, 1).view(n_heads, -1, embed_dim).transpose(1, 2)
            model[p_template.format(coder, i, "out", "weight")] = out_weight
            model[p_template.format(coder, i, "out", "bias")] = out_bias.unsqueeze(0)

        return model


def main(args):
    state = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint)
    ns = state["args"]
    model = state["model"]
    ns.arch = "transformer_modular"

    if (
            args.encoder_attention_heads_active is None 
            and args.decoder_attention_heads_active is None):
        raise ValueError(
            'Either --encoder-attention-heads-active or '
            '--decoder-attention-heads-active option must be set.')
    if args.encoder_attention_heads_active is None:
        args.encoder_attention_heads_active = args.decoder_attention_heads_active

    if args.encoder_modular_layer_indices is not None:
        ns.encoder_modular_layer_indices = "({})".format(args.encoder_modular_layer_indices)
        model = convert_model(model, ns, coder="encoder", att_type="self_attn")
    if args.decoder_modular_layer_indices is not None:
        ns.decoder_modular_layer_indices = "({})".format(args.decoder_modular_layer_indices)
        model = convert_model(model, ns, coder="decoder", att_type="self_attn")
        model = convert_model(model, ns, coder="decoder", att_type="encoder_attn")

    ctrl_enc = ModularCtrl(
        ns.encoder_embed_dim,
        ns.encoder_attention_heads,
        args.encoder_attention_heads_active,
        hidden_depth=args.ctrl_hidden_depth,
        hidden_dim=args.ctrl_hidden_dim,
        ctrl_type=args.ctrl_type)
    ns.module_ctrl_hidden_depth = args.ctrl_hidden_depth
    ns.module_ctrl_hidden_dim = args.ctrl_hidden_dim
    ns.module_ctrl_type = args.ctrl_type

    for k, v in ctrl_enc.state_dict().items():
        model["encoder.module_ctrl.{}".format(k)] = v

    if not args.share_encoder_ctrl:
        if args.decoder_attention_heads_active is None:
            raise ValueError(
                "Missing ``decoder-attention-heads-active'' "
                "when ``share-encoder-ctrl'' is disabled.")
        ns.share_encoder_ctrl = False
        ctrl_dec = ModularCtrl(
            ns.decoder_embed_dim,
            ns.decoder_attention_heads,
            args.decoder_attention_heads_active,
            hidden_depth=args.ctrl_hidden_depth,
            hidden_dim=args.ctrl_hidden_dim,
            ctrl_type=args.ctrl_type)
        for k, v in ctrl_dec.state_dict().items():
            model["decoder.module_ctrl.{}".format(k)] = v
    else:
        ns.share_encoder_ctrl = True

    ns.arch = "transformer_modular"
    ns.criterion = "label_smoothed_cross_entropy_modular"
    ns.task = "translation_modular"
    ns.encoder_attention_heads_active = args.encoder_attention_heads_active

    state["args"] = ns
    state["model"] = model

    for i, _ in enumerate(state["optimizer_history"]):
        state["optimizer_history"][i]["criterion_name"] = 'LabelSmoothedCrossEntropyModularCriterion'

    state = utils.move_to_cpu(state)

    with PathManager.open(args.save_as, "wb") as f:
        checkpoint_utils.torch_persistent_save(state, f)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True)
    parser.add_argument(
        "--encoder-modular-layer-indices", type=str, default=None)
    parser.add_argument(
        "--encoder-attention-heads-active", type=int, default=None)
    parser.add_argument(
        "--decoder-modular-layer-indices", type=str, default=None)
    parser.add_argument(
        "--decoder-attention-heads-active", type=int, default=None)
    parser.add_argument(
        "--ctrl-type", type=str, default="joint")
    parser.add_argument(
        "--ctrl-hidden-depth", type=int, default=3)
    parser.add_argument(
        "--ctrl-hidden-dim", type=int, default=512)
    parser.add_argument(
        "--share-encoder-ctrl", action="store_true")
    parser.add_argument(
        "--save-as", type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
