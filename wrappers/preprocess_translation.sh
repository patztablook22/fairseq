#!/bin/bash

EXPDIR=$1
SRC=$2
TGT=$3
TEXT=${4:="examples/translation/iwslt14.tokenized.de-en"}
VALID_SETS=${5:-"valid"}
TEST_SET=${6:-""}

echo $VALID_SETS | sed "s#^#$TEXT/#;s#,#,$TEXT/#g"

TEST_SET_OPT=""
[[ -n $TEST_SET ]] && TEST_SET_OPT="--testpref $TEXT/$TEST_SET"
echo $TEST_SET_OPT

fairseq-preprocess \
    --source-lang $SRC \
    --target-lang $TGT \
    --trainpref $TEXT/train \
    --validpref `echo $VALID_SETS | sed "s#^#$TEXT/#;s#,#,$TEXT/#g"` \
    $TEST_SET_OPT \
    --destdir $EXPDIR/data \
    --workers 20
