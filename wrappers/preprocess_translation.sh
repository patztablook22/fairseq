#!/bin/bash
set -e

WORKERS=10

EXP_DIR=

SRC=de
TGT=en

DATA_DIR="custom_examples/translation/iwslt14_deen"
TRAIN_SET="iwslt14"
VALID_SETS="newstest iwslt17_tst"
TEST_SETS="newstest iwslt17_tst"

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

LANG=$SRC-$TGT

mkdir -p $EXP_DIR/data
python preprocess.py \
    --source-lang $SRC \
    --target-lang $TGT \
    --trainpref $DATA_DIR/$TRAIN_SET.train \
    --destdir $EXP_DIR/data \
    $JOINED_DICT_OPT \
    --workers $WORKERS
for suf in bin idx; do
    for l in $SRC $TGT; do
        mv $EXP_DIR/data/train.$LANG.$l.$suf \
            $EXP_DIR/data/$TRAIN_SET.train.$LANG.$l.$suf
    done
done

for dataset in $VALID_SETS; do
    python preprocess.py \
        --source-lang $SRC \
        --target-lang $TGT \
        --validpref $DATA_DIR/$dataset.valid \
        --destdir $EXP_DIR/data \
        --srcdict $EXP_DIR/data/dict.$SRC.txt \
        --tgtdict $EXP_DIR/data/dict.$TGT.txt \
        --workers $WORKERS
    for suf in bin idx; do
        for l in $SRC $TGT; do
            mv $EXP_DIR/data/valid.$LANG.$l.$suf \
                $EXP_DIR/data/$dataset.valid.$LANG.$l.$suf
        done
    done
done

for dataset in $TEST_SETS; do
    python preprocess.py \
        --source-lang $SRC \
        --target-lang $TGT \
        --testpref $DATA_DIR/$dataset.test \
        --destdir $EXP_DIR/data \
        --srcdict $EXP_DIR/data/dict.$SRC.txt \
        --tgtdict $EXP_DIR/data/dict.$TGT.txt \
        --workers $WORKERS
    for suf in bin idx; do
        for l in $SRC $TGT; do
            mv $EXP_DIR/data/test.$LANG.$l.$suf \
                $EXP_DIR/data/$dataset.test.$LANG.$l.$suf
        done
    done
done
