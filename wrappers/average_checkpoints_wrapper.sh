#!/bin/bash

EXPDIR=$1
N_CHECKPOINTS=5

python scripts/average_checkpoints.py \
    --inputs $EXPDIR/checkpoints \
    --num-epoch-checkpoints $N_CHECKPOINTS \
    --output $EXPDIR/checkpoint.avg$N_CHECKPOINTS.pt
