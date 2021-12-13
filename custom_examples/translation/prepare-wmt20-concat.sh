#!/bin/bash
WD=`pwd`

SRC=en
TGT=cs

MIN_LEN=50
MAX_LEN=60
INPUT_LENGTHS="10 20 30"

ORIG_DIR=${1:-"$WD/wmt20_${SRC}${TGT}"}
OUTDIR=$ORIG_DIR.concat.$MAX_LEN

[[ -d $ORIG_DIR ]] || echo $ORIG_DIR does not exist. Please run prepare-wmt20.sh to creat $ORIG_DIR

SCRIPTS=$WD/mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl


mkdir $OUTDIR
ln -s $ORIG_DIR/bpecodes $OUTDIR/.
ln -s $ORIG_DIR/czeng.train.* $OUTDIR/.
ln -s $ORIG_DIR/newstest.valid.* $OUTDIR/.
ln -s $ORIG_DIR/newstest.test.* $OUTDIR/.

for l in 10 20 30; do
    ~/scripts/combine_sentpairs_by_length.py \
        --input-file $ORIG_DIR/bpe.$l.train \
        --min-length $MIN_LEN \
        --max-length $MAX_LEN \
        --field tgt > $OUTDIR/bpe.$l.train
    cut -f1 $OUTDIR/bpe.$l.train > $OUTDIR/bpe.$l.train.$SRC
    cut -f2 $OUTDIR/bpe.$l.train > $OUTDIR/bpe.$l.train.$TGT

    ln -s $OUTDIR/bpe.$l.train.$SRC $OUTDIR/czeng.$l.train.$SRC
    ln -s $OUTDIR/bpe.$l.train.$TGT $OUTDIR/czeng.$l.train.$TGT

    for t in "valid" "test"; do
        ~/scripts/combine_sentpairs_by_length.py \
            --input-file $ORIG_DIR/bpe.newstest.$l.$t \
            --min-length $MIN_LEN \
            --max-length $MAX_LEN \
            --field tgt > $OUTDIR/bpe.newstest.$l.$t
        cut -f1 $OUTDIR/bpe.newstest.$l.$t | \
            sed 's/@@ //g' | \
            perl $DETOKENIZER -l $SRC \
            > $OUTDIR/newstest.$l.$t.$SRC
        cut -f2 $OUTDIR/bpe.newstest.$l.$t | \
            sed 's/@@ //g' | \
            perl $DETOKENIZER -l $TGT \
            > $OUTDIR/newstest.$l.$t.$TGT
    done
done
