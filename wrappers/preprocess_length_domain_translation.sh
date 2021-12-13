#!/bin/bash
set -e

LENGTHS="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
WORKERS=4

EXP_DIR=

SRC=en
TGT=cs

DATA_DIR="custom_examples/translation/wmt20_${SRC}${TGT}"
TRAIN_SET="czeng"
VALID_SET="newstest"
TEST_SET="newstest"

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
    --train-set)
        TRAIN_SET="$2"
        shift
    ;;
    --valid-set)
        VALID_SET="$2"
        shift
    ;;
    --test-set)
        TEST_SET="$2"
        shift
    ;;
    --lengths)
        LENGTHS="$2"
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

LANG=$SRC-$TGT

mkdir -p $EXP_DIR/data
python preprocess.py \
    --source-lang $SRC \
    --target-lang $TGT \
    --trainpref $DATA_DIR/$TRAIN_SET.train \
    --validpref $DATA_DIR/$VALID_SET.valid \
    --testpref $DATA_DIR/$TEST_SET.test \
    --destdir $EXP_DIR/data \
    $JOINED_DICT_OPT \
    --workers $WORKERS
for suf in bin idx; do
    for l in $SRC $TGT; do
        mv $EXP_DIR/data/train.$LANG.$l.$suf \
            $EXP_DIR/data/$TRAIN_SET.train.$LANG.$l.$suf
    done
done
for suf in bin idx; do
    for l in $SRC $TGT; do
        mv $EXP_DIR/data/valid.$LANG.$l.$suf \
            $EXP_DIR/data/$VALID_SET.valid.$LANG.$l.$suf
        mv $EXP_DIR/data/test.$LANG.$l.$suf \
            $EXP_DIR/data/$TEST_SET.test.$LANG.$l.$suf
    done
done

for len in $LENGTHS; do
    python preprocess.py \
        --source-lang $SRC \
        --target-lang $TGT \
        --trainpref $DATA_DIR/$TRAIN_SET.$len.train \
        --destdir $EXP_DIR/data \
        --srcdict $EXP_DIR/data/dict.$SRC.txt \
        --tgtdict $EXP_DIR/data/dict.$TGT.txt \
        --workers $WORKERS
    for suf in bin idx; do
        for l in $SRC $TGT; do
            mv $EXP_DIR/data/train.$LANG.$l.$suf \
                $EXP_DIR/data/czeng.$len.train.$LANG.$l.$suf
        done
    done
done
