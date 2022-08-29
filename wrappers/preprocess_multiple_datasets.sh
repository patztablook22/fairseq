#!/bin/bash
set -e

WORKERS=4

EXP_DIR=

SRC=en
TGT=cs

DATA_DIR="custom_examples/translation/bitedit.30000"
TRAIN_SETS="all.15"
VALID_SETS="id.15 push.15 pop.15 shift.15 unshift.15 reverse.15"
TEST_SETS="id.15 push.15 pop.15 shift.15 unshift.15 reverse.15"

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
    --train-sets)
        TRAIN_SETS="$2"
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
# Concatenate train sets to create single vocabulary
for l in $SRC $TGT; do
    [[ -e $EXP_DIR/data/concatenated.train.$l ]] && rm $EXP_DIR/data/concatenated.train.$l
    for d in $TRAIN_SETS; do
        cat $DATA_DIR/$d.train.$l >> $EXP_DIR/data/concatenated.train.$l
    done
done

# Create the vocabularies and binarize valid/test data
python preprocess.py \
    --source-lang $SRC \
    --target-lang $TGT \
    --trainpref $EXP_DIR/data/concatenated.train \
    --validpref $(echo $VALID_SETS | sed "s#^\([^ ]*\)#$DATA_DIR/\1.valid#;s# \([^ ]*\)#,$DATA_DIR/\1.valid#g") \
    --testpref $(echo $TEST_SETS | sed "s#^\([^ ]*\)#$DATA_DIR/\1.test#;s# \([^ ]*\)#,$DATA_DIR/\1.test#g") \
    --destdir $EXP_DIR/data \
    $JOINED_DICT_OPT \
    --workers $WORKERS
for suf in bin idx; do
    for l in $SRC $TGT; do
        # rename valid/test (0) files first
        for t in valid test; do 
            mv $EXP_DIR/data/$t.$LANG.$l.$suf \
                $EXP_DIR/data/${t}0.$LANG.$l.$suf
        done

        i=0
        for s in $VALID_SETS; do
            mv $EXP_DIR/data/valid${i}.$LANG.$l.$suf \
                $EXP_DIR/data/$s.valid.$LANG.$l.$suf
            i=$(expr $i + 1)
        done
        i=0
        for s in $TEST_SETS; do
            mv $EXP_DIR/data/test${i}.$LANG.$l.$suf \
                $EXP_DIR/data/$s.test.$LANG.$l.$suf
            i=$(expr $i + 1)
        done
    done
done

# We don't keep the concatenated dataset (only the dicts)
rm $EXP_DIR/data/concatenated.train.*

for d in $TRAIN_SETS; do
    python preprocess.py \
        --source-lang $SRC \
        --target-lang $TGT \
        --trainpref $DATA_DIR/$d.train \
        --destdir $EXP_DIR/data \
        --srcdict $EXP_DIR/data/dict.$SRC.txt \
        --tgtdict $EXP_DIR/data/dict.$TGT.txt \
        --workers $WORKERS
    for suf in bin idx; do
        for l in $SRC $TGT; do
            mv $EXP_DIR/data/train.$LANG.$l.$suf \
                $EXP_DIR/data/$d.train.$LANG.$l.$suf
        done
    done
done
