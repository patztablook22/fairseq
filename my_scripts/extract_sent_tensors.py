#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.models.transformer_modular import TransformerModularModel
from fairseq.data import encoders
from fairseq_cli.interactive import buffered_read, make_batches


def main(args):
    ckpt = torch.load(args.checkpoint)
    ckpt['args'].path = args.checkpoint
    task = tasks.setup_task(ckpt['args'])

    models, _model_args = checkpoint_utils.load_model_ensemble(
        ckpt['args'].path.split(os.pathsep),
        arg_overrides=eval("{}"),
        task=task,
    )
    model = models[0]

    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    tokenizer = encoders.build_tokenizer(ckpt['args'])
    bpe = encoders.build_bpe(ckpt['args'])

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    start_id = 0
    outputs = {
        "id": [],
        "enc_ctrl_feat": [],
        "dec_ctrl_feat": [],
    }
    ctrl_enc = getattr(model.encoder, 'module_ctrl', None)
    ctrl_dec = getattr(model.decoder, 'module_ctrl', None)
    if ctrl_enc is not None:
        outputs['enc_ctrl_pred'] = []
    if ctrl_dec is not None:
        outputs['dec_ctrl_pred'] = []
    for inputs in buffered_read(args.input_file, 100):
        results = []
        with torch.no_grad():
            for batch in make_batches(inputs, ckpt['args'], task, max_positions, encode_fn):
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                #if use_cuda:
                #    src_tokens = src_tokens.cuda()
                #    src_lengths = src_lengths.cuda()

                bsz = src_tokens.size(0)
                mask = src_tokens.eq(
                    model.encoder.padding_idx).view(bsz, -1, 1)

                emb_out, _ = model.encoder.forward_embedding(src_tokens)

                if ctrl_enc is not None:
                    enc_ctrl_feat = ctrl_enc.extract_features(
                        emb_out.clone(),
                        padding_mask=mask)
                    enc_ctrl_out = ctrl_enc(
                        emb_out.clone(),
                        'validation',
                        padding_mask=mask)

                enc_out = model.encoder.forward(src_tokens, src_lengths)
                enc_out = enc_out.encoder_out.transpose(0, 1)
                if ctrl_dec is not None:
                    dec_ctrl_feat = ctrl_dec.extract_features(
                        enc_out.clone(),
                        padding_mask=mask)
                    dec_ctrl_out = ctrl_dec(
                        enc_out.clone(),
                        'validation',
                        padding_mask=mask
                    )

                for i, idx in enumerate(batch.ids.tolist()):
                    outputs['id'].append(idx)
                    outputs['enc_ctrl_feat'].append(enc_ctrl_feat[i].numpy())
                    outputs['dec_ctrl_feat'].append(dec_ctrl_feat[i].numpy())
                    if ctrl_enc is not None:
                        outputs['enc_ctrl_pred'].append(
                            enc_ctrl_out.ctrl_prediction[i].numpy())
                    if ctrl_dec is not None:
                        outputs['dec_ctrl_pred'].append(
                            dec_ctrl_out.ctrl_prediction[i].numpy())

    for k in outputs:
        outputs[k] = np.stack(outputs[k], 0)
    if ctrl_enc is not None:
        outputs['enc_ctrl_proj'] = ctrl_enc.out_proj.weight.detach().numpy()
    if ctrl_dec is not None:
        outputs['dec_ctrl_proj'] = ctrl_dec.out_proj.weight.detach().numpy()

    np.savez(args.output_file, **outputs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, required=True)
    parser.add_argument(
        "--input-file", type=str, required=True)
    parser.add_argument(
        "--output-file", type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
