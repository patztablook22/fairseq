#!/bin/bash
CZENG=$1
BUCKET_CRITERION=${2:-"tgt"}  # criterions: "tgt", "src", "both"

WD=`pwd`

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

echo 'Cloning SacreBLEU repository (for newstest data download)'
git clone https://github.com/mjpost/sacrebleu.git

SCRIPTS=$WD/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=$WD/subword-nmt/subword_nmt
SENT_FILTER=$WD/../../my_scripts/filter_sentpairs_by_length.py

SRC=en
TGT=cs
LANG=$SRC-$TGT

BPE_TOKENS=30000
BUCKETS="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"

OUTDIR=$WD/wmt20_${SRC}${TGT}.$BUCKET_CRITERION
TMPDIR=$OUTDIR/tmp.d

mkdir -p $OUTDIR
mkdir -p $TMPDIR

# Prepare CzEng20
echo "Preparing CzEng20 train data..."
zcat $CZENG/czeng20-train.gz | cut -f6 | \
    perl $NORM_PUNC | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -a -l en > $TMPDIR/train.tok.en
zcat $CZENG/czeng20-train.gz | cut -f5 | \
    perl $NORM_PUNC | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -a -l cs > $TMPDIR/train.tok.cs

TRAIN=$TMPDIR/train
BPE_CODE=$OUTDIR/bpecodes
rm -f $TRAIN
for l in $SRC $TGT; do
    cat $TMPDIR/train.tok.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for l in $SRC $TGT; do
    f_in=$TMPDIR/train.tok.$l
    f_out=$OUTDIR/bpe.train.$l

    echo "apply_bpe.py to ${f_in}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f_in > $f_out
    ln -s $OUTDIR/bpe.train.$l $OUTDIR/czeng.train.$l
done

# Download WMT newstests
cd $OUTDIR
echo "preparing valid/test data..."
for y in 13 14 15 16 17 18 19 20; do
    sacrebleu -t wmt$y -l $LANG --echo src > $TMPDIR/newstest$y.$SRC
    sacrebleu -t wmt$y -l $LANG --echo ref > $TMPDIR/newstest$y.$TGT

    for l in $SRC $TGT; do
        f_in=$TMPDIR/newstest$y.$l
        f_out=$TMPDIR/bpe.newstest$y.$l

        echo "pre-processing ${f_in}..."
        cat $f_in | \
            perl $NORM_PUNC | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -a -l $l | \
            python $BPEROOT/apply_bpe.py -c $BPE_CODE > $f_out
    done
done

for l in $SRC $TGT; do
    f_out=$OUTDIR/bpe.newstest.valid.$l
    rm $f_out
    for y in 13 14 15 16; do
        cat $TMPDIR/bpe.newstest$y.$l >> $f_out
    done
    ln -s $OUTDIR/bpe.newstest.valid.$l $OUTDIR/newstest.valid.$l

    f_out=$OUTDIR/bpe.newstest.test.$l
    rm $f_out
    for y in 17 18 19 20; do
        cat $TMPDIR/bpe.newstest$y.$l >> $f_out
    done
    ln -s $OUTDIR/bpe.newstest.test.$l $OUTDIR/newstest.test.$l
done

min_len=0
echo "splitting data to length buckets..."
for max_len in $BUCKETS; do
    echo "processing bucket length ${max_len}..."
    paste $OUTDIR/bpe.train.$SRC $OUTDIR/bpe.train.$TGT | \
        grep -v "^[^[:alnum:]]$" | \
        $SENT_FILTER \
            --min-length $min_len \
            --max-length $max_len \
            --field $BUCKET_CRITERION > $OUTDIR/bpe.$max_len.train
    cut -f1 $OUTDIR/bpe.$max_len.train \
        > $OUTDIR/bpe.$max_len.train.$SRC
    cut -f2 $OUTDIR/bpe.$max_len.train \
        > $OUTDIR/bpe.$max_len.train.$TGT

    ln -s $OUTDIR/bpe.$max_len.train.$SRC \
        $OUTDIR/czeng.$max_len.train.$SRC
    ln -s $OUTDIR/bpe.$max_len.train.$TGT \
        $OUTDIR/czeng.$max_len.train.$TGT

    for t in "valid" "test"; do
        paste $OUTDIR/bpe.newstest.$t.$SRC $OUTDIR/bpe.newstest.$t.$TGT | \
            grep -v "^[^[:alnum:]]$" | \
            $SENT_FILTER \
                --min-length $min_len \
                --max-length $max_len \
                --field $BUCKET_CRITERION > $OUTDIR/bpe.newstest.$max_len.$t
        cut -f1 $OUTDIR/bpe.newstest.$max_len.$t | \
            sed 's/@@ //g' | \
            perl $DETOKENIZER -l $SRC \
            > $OUTDIR/newstest.$max_len.$t.$SRC
        cut -f2 $OUTDIR/bpe.newstest.$max_len.$t | \
            sed 's/@@ //g' | \
            perl $DETOKENIZER -l $TGT \
            > $OUTDIR/newstest.$max_len.$t.$TGT
    done
    min_len=$max_len
done

rm -r $TMPDIR
