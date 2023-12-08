#!/bin/bash


INPUT=$1
EXPDIR=$2
CKPT_ID=${3:-"_best"}
PREFIX=${4:-"results"}
OPTS="-s x -t y"

BEAM_SIZE=1
LP=0.6

#CKPT_ID="_last"
#CKPT_ID="_best"
#CKPT_ID="1"

DATA_PATH=$EXPDIR/../data
RESULTS_FILE=$PREFIX.${CKPT_ID##"_"}.txt
CKPT=$EXPDIR/checkpoints/checkpoint$CKPT_ID.pt

echo .$INPUT
echo .

python interactive.py \
    $DATA_PATH \
    --input $INPUT \
    --path $CKPT \
    --beam $BEAM_SIZE \
    --lenpen $LP \
    --remove-bpe \
    --buffer-size 500 \
    $OPTS | tee $RESULTS_FILE
