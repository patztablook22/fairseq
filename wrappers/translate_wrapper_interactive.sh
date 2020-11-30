#!/bin/bash

EXPDIR=$1
CKPT_ID=${2:-"_best"}
PREFIX=${3:-"results"}
#OPTS=${4:-"--print-selection --print-attn-confidence"}
OPTS=""

#CKPT_ID="_last"
#CKPT_ID="_best"
#CKPT_ID="1"

INPUT_PATH=$EXPDIR/../data
RESULTS_FILE=$EXPDIR/$PREFIX.${CKPT_ID##"_"}.txt
CKPT=$EXPDIR/checkpoints/checkpoint$CKPT_ID.pt

cat /dev/stdin | fairseq-interactive \
    $INPUT_PATH \
    --path $CKPT \
    --beam 4 \
    --lenpen 0.6 \
    --remove-bpe \
    --buffer-size 500 \
    $OPTS | tee $RESULTS_FILE
