#!/bin/bash

set -ex

FILE=$1
N_HEADS=$2
N_MODULES=$3
TYPE="joint"

SRC="de"
TGT="en"

mod_stats_opt=""
if [[ $TYPE == "joint" ]]; then
    mod_stats_opt="--n-modules $N_MODULES --n-active $N_HEADS"
fi


SUBSET_FILE=${FILE}.all
echo "Processing ${SUBSET_FILE}" >&2
cat $FILE | \
    grep '^.-' | \
    tr "\t" "@" | \
    paste - - - - - - - | \
    sed 's/\t.-[^@]*@/\t/g' | \
    sed 's/^\([^@]*\)@/\1\t/' > ${SUBSET_FILE}.tsv
cut -f1,7 ${SUBSET_FILE}.tsv | \
    ./print_modular_statistics.py $mod_stats_opt > ${SUBSET_FILE}.mod_stats
cut -f7,8 ${SUBSET_FILE}.tsv > ${SUBSET_FILE}.mod_conf


cut -f2 ${SUBSET_FILE}.tsv | \
    ~/scripts/udpipe_wrapper.sh $SRC \
        --tag \
        --parse \
        --input=horizontal > ${SUBSET_FILE}.conll

cut -f3 ${SUBSET_FILE}.conll | \
    grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.lemma
~/scripts/analyze_corpus.sh \
    -l $SRC \
    --input ${SUBSET_FILE}.lemma 

cut -f4 ${SUBSET_FILE}.conll | \
    grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.pos
~/scripts/analyze_corpus.sh \
    -l $SRC \
    --input ${SUBSET_FILE}.pos 

cut -f6 ${SUBSET_FILE}.conll | \
    grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.morpho
~/scripts/analyze_corpus.sh \
    -l $SRC \
    --input ${SUBSET_FILE}.morpho 

cut -f8 ${SUBSET_FILE}.conll | \
    grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.syntax
~/scripts/analyze_corpus.sh \
    -l $SRC \
    --input ${SUBSET_FILE}.syntax


for m in `cat ${FILE}.all.tsv | cut -f7 | sort -u`; do
    SUBSET_FILE=${FILE}.${m}
    echo "Processing ${SUBSET_FILE}" >&2
    grep -P "\t${m}\t" ${FILE}.all.tsv > ${SUBSET_FILE}.tsv
    cut -f1 ${SUBSET_FILE}.tsv > ${SUBSET_FILE}.ids

    cut -f2 ${SUBSET_FILE}.tsv | \
        ~/scripts/udpipe_wrapper.sh $SRC \
            --tag \
            --parse \
            --input=horizontal > ${SUBSET_FILE}.conll

    cut -f3 ${SUBSET_FILE}.conll | \
        grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.lemma
    ~/scripts/analyze_corpus.sh \
        -l $SRC \
        --input ${SUBSET_FILE}.lemma 

    cut -f4 ${SUBSET_FILE}.conll | \
        grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.pos
    ~/scripts/analyze_corpus.sh \
        -l $SRC \
        --input ${SUBSET_FILE}.pos 

    cut -f6 ${SUBSET_FILE}.conll | \
        grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.morpho
    ~/scripts/analyze_corpus.sh \
        -l $SRC \
        --input ${SUBSET_FILE}.morpho 

    cut -f8 ${SUBSET_FILE}.conll | \
        grep -v '^#' | tr "\n" " " | sed 's/  /\n/g' > ${SUBSET_FILE}.syntax
    ~/scripts/analyze_corpus.sh \
        -l $SRC \
        --input ${SUBSET_FILE}.syntax
done
