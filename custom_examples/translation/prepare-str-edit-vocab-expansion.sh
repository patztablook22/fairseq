#!/usr/bin/env bash
set -e
WD=`pwd`

GENERATOR="../../my_scripts/create_bitedit_examples.py"
SIZE=${1:-30000}
OP=${2:-"id"}
CHARPAIRS=${3:-"a-b c-d a-c b-d"}

OUTDIR=$WD/str-edit.vocab.$OP.$SIZE
TMPDIR=$OUTDIR/tmp.d

mkdir -p $OUTDIR $TMPDIR

for pair in $CHARPAIRS; do
    zero_char=`echo $pair | cut -d'-' -f1`
    one_char=`echo $pair | cut -d'-' -f2`

    old_n=1
    for n in 20 30 40 50 60; do
        tmp_out=$TMPDIR/$pair.$n
        echo "Creating '$tmp_out' dataset..." >&2

        $GENERATOR \
            --seed 42 \
            --task $OP \
            --min-n-bits $old_n \
            --max-n-bits $n \
            --n-examples $SIZE \
            --zero-char $zero_char \
            --one-char $one_char \
            > $tmp_out

        awk '{if (NR <= 1000) print $0;}' $tmp_out > $tmp_out.valid
        awk '{if (NR > 1000 && NR <= 2000) print $0;}' $tmp_out > $tmp_out.test
        awk '{if (NR > 2000) print $0;}' $tmp_out > $tmp_out.train

        for t in train valid test; do
            f_in=$TMPDIR/$pair.$n.$t
            f_out=$OUTDIR/$pair.$n.$t
            cut -f2 $f_in > $f_out.x
            cut -f3 $f_in > $f_out.y
        done

        old_n=$(expr $n + 1)
    done
done

for t in train valid test; do
    for n in 20 30 40 50 60; do
        f_in=$TMPDIR/all.$n.$t
        f_out=$OUTDIR/all.$n.$t
        echo "Creating '$f_out' dataset..." >&2

        cat $TMPDIR/*.$n.$t | shuf > $f_in
        cut -f2 $f_in > $f_out.x
        cut -f3 $f_in > $f_out.y
    done
done

rm -r $TMPDIR
