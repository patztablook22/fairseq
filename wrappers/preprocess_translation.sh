#!/bin/bash
set -e

WORKERS=20

EXP_DIR=

SRC=de
TGT=en

DATA_DIR="custom_examples/translation/iwslt14.tokenized.de-en"
VALID_SETS="valid"
TEST_SETS="test"

JOINED_DICT_OPT=


HELP=1
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --expdir)
        EXP_DIR="$2"
        shift
    ;;
    --src)
        SRC="$2"
        shift
    ;;
    --tgt)
        TGT="$2"
        shift
    ;;
    --datadir)
        DATA_DIR="$2"
        shift
    ;;
    --valid-sets)
        VALID_SETS="$2"
        shift
    ;;
    --test-sets)
        TEST_SETS="$2"
        shift
    ;;
    --joined-dict)
        JOINED_DICT_OPT="--joined-dictionary"
    ;;
    -h|--help)
        HELP=0
    ;;
    *)
        echo Unknown option '"'$key'"' >&2
        exit 1
    ;;
esac
shift
done

VALID_SETS=`echo $VALID_SETS | sed "s#^#$DATA_DIR/#;s#,#,$DATA_DIR/#g"`
TEST_SETS=`echo $TEST_SETS | sed "s#^#$DATA_DIR/#;s#,#,$DATA_DIR/#g"`

VALID_SET_OPT=""
[[ -n $VALID_SETS ]] && VALID_SET_OPT="--validpref $VALID_SETS"
TEST_SET_OPT=""
[[ -n $TEST_SETS ]] && TEST_SET_OPT="--testpref $TEST_SETS"

mkdir $EXP_DIR/data
fairseq-preprocess \
    --source-lang $SRC \
    --target-lang $TGT \
    --trainpref $DATA_DIR/train \
    $VALID_SET_OPT \
    $TEST_SET_OPT \
    --destdir $EXP_DIR/data \
    $JOINED_DICT_OPT \
    --workers $WORKERS
