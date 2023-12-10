#!/bin/bash

cd "$(dirname $0)/.."

wrappers/train_transformer.sh \
    -e $SCRATCH/experiments/str-edit.compo.30000 \
    --eval-dir custom_examples/translation/str-edit.compo.30000 \
    --emb-size 128 \
    --ffn-size "$(expr 4 \* 128)" \
    --depth 1 \
    --att-heads 1 \
    --lr 1e-4 \
    --epochs-not-updates \
    --max-updates 100 \
    --minimize-metric \
    --best-metric ter \
    2>&1
