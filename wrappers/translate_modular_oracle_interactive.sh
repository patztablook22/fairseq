#!/bin/bash

EXPDIR=$1
INPUT_FILE=$2
CKPT_ID=${3:-"_best"}
PREFIX=${4:-"oracle"}
OPTS=${5:-"--print-selection --print-attn-confidence"}

#CKPT_ID="_last"
#CKPT_ID="_best"
#CKPT_ID="1"

INPUT_PATH=$EXPDIR/../data
RESULTS_FILE=$EXPDIR/$PREFIX.${CKPT_ID##"_"}.txt
CKPT=$EXPDIR/checkpoints/checkpoint$CKPT_ID.pt

python fairseq_cli/modular_oracle.py \
    $INPUT_PATH \
    --input $INPUT_FILE \
    --task "translation_modular" \
    --path $CKPT \
    --beam 4 \
    --lenpen 0.6 \
    --remove-bpe \
    --buffer-size 500 \
    $OPTS | tee $RESULTS_FILE
