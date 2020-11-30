#!/bin/bash

EXPDIR=$1
CKPT_ID=${2:-"_best"}
OPTS=${3:-"--print-selection --print-attn-confidence"}

#CKPT_ID="_last"
#CKPT_ID="_best"
#CKPT_ID="1"

INPUT_PATH=$EXPDIR/../data
RESULTS_FILE=$EXPDIR/results.${CKPT_ID##"_"}.txt
CKPT=$EXPDIR/checkpoints/checkpoint$CKPT_ID.pt

fairseq-generate \
    $INPUT_PATH \
    --path $CKPT \
    --beam 4 \
    --lenpen 0.6 \
    --remove-bpe \
    $OPTS | tee $RESULTS_FILE
