#!/bin/bash

EXP_DIR=$1
shift
YEAR=$1
shift

PREF="1-head"
TYPE=".oracle"

SRC_LANG=de
TGT_LANG=en

IWSLT_REF_ROOT="$HOME/tspec-workdir/fairseq/custom_examples/translation/iwslt14.tokenized.de-en.iwslt17-validation"

mkdir -p tmp.d
mkdir -p multeval_out
rm tmp.d/multeval_iwslt*

MULTEVAL_OPT=""
i=1
for suf in $@; do
    for sys in $EXP_DIR/*$suf; do
        grep ^H- $sys/iwslt17_tst${YEAR}$TYPE.best.txt | cut -f3 > tmp.d/multeval_iwslt_$i.$RANDOM
    done
    MULTEVAL_OPT="$MULTEVAL_OPT --hyps-sys$i `echo tmp.d/multeval_iwslt_$i.*`"
    i=$((i+1))
done

BASE_DIR="$HOME/tspec-workdir/fairseq/experiments/iwslt14.tokenized.de-en.clusters/baseline"
BASE_SUFF="warmup-4000.clip-norm-0.0.emb-size-512.dec-att-heads-8.enc-att-heads-8.lr-5e-4.max-tokens-4096"
rm tmp.d/multeval_base_iwslt*
for sys in $BASE_DIR/*$BASE_SUFF; do
    grep ^H- $sys/iwslt17_tst$YEAR.best.txt | cut -f3 > tmp.d/multeval_base_iwslt.$RANDOM
done

MULTEVAL_OPT=`echo $MULTEVAL_OPT | sed 's/^ //'`

/home/varis/tspec-workdir/multeval-0.5.1/multeval.sh eval \
    --refs $IWSLT_REF_ROOT/iwslt17_tst$YEAR.tok.lc.bpe.$SRC_LANG-$TGT_LANG.$TGT_LANG \
    --hyps-baseline tmp.d/multeval_base_iwslt* \
    $MULTEVAL_OPT \
    --latex multeval_out/${PREF}${TYPE}.$YEAR.tex \
    --rankDir tmp.d/${PREF}${TYPE}.$YEAR.rank \
    --meteor.language en
