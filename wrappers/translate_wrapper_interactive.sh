#!/bin/bash

EXPDIR=$1
CKPT_ID=${2:-"_best"}
PREFIX=${3:-"$EXPDIR/results"}
#OPTS=${4:-"--print-selection --print-attn-confidence"}
OPTS=${4:-"--beam 4 --lenpen 0.6"}

#CKPT_ID="_last"
#CKPT_ID="_best"
#CKPT_ID="1"

INPUT_PATH=$EXPDIR/../data
RESULTS_FILE=$PREFIX.${CKPT_ID##"_"}.txt
CKPT=$EXPDIR/checkpoints/checkpoint$CKPT_ID.pt

#cat /dev/stdin | cuda-memcheck python interactive.py \
cat /dev/stdin | python interactive.py \
    $INPUT_PATH \
    --path $CKPT \
    --buffer-size 500 \
    $OPTS | tee $RESULTS_FILE
