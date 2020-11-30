#!/bin/bash

EXPDIR=$1
SRCS=$2
TGT=$3
TEXT=${4:-"examples/translation/iwslt17.de_fr.en.bpe16k"}
VALID_SETS=${5:-"valid0.bpe,valid1.bpe,valid2.bpe,valid3.bpe,valid4.bpe,valid5.bpe"}


for SRC in `echo $SRCS | tr "," " "`; do
    OPTS=""
    [[ -e $EXPDIR/data/dict.$TGT.txt ]] && OPTS="--tgtdict $EXPDIR/data/dict.$TGT.txt"

    fairseq-preprocess \
        --source-lang $SRC \
        --target-lang $TGT \
        --trainpref $TEXT/train.bpe.$SRC-$TGT \
        --validpref `echo $VALID_SETS | sed "s/bpe/bpe.$SRC-$TGT/g;s#valid#$TEXT/valid#g"` \
        --destdir $EXPDIR/data \
        --workers 20 \
        $OPTS
done
