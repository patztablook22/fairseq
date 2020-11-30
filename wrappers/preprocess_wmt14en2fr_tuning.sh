#!/bin/bash

EXPDIR=$1
TEXT=${2:-"custom_examples/translation/wmt14_enfr_tuning"}

SRC="en"
TGT="fr"

fairseq-preprocess \
    -s $SRC \
    -t $TGT \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --srcdict $EXPDIR/data/dict.$SRC.txt \
    --tgtdict $EXPDIR/data/dict.$TGT.txt \
    --destdir $EXPDIR/data \
    --workers 20
