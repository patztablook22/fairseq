#!/usr/bin/env bash
set -e
WD=`pwd`

GENERATOR="../../my_scripts/create_bitedit_examples.py"
SIZE=${1:-30000}
OP=${2:-"id"}
CHARPAIRS=${3:-"a-b c-d a-c b-d"}

OUTDIR=$WD/string-edit.$OP.$SIZE
TMPDIR=$OUTDIR/tmp.d

mkdir -p $OUTDIR $TMPDIR

old_n=1
for pair in $CHARPAIRS; do
    zero_char=`echo $pair | cut -d'-' -f1`
    one_char=`echo $pair | cut -d'-' -f2`
    for n in 15 20 25 30 35 40; do
        echo "Creating '$pair.$n' dataset..." >&2
        tmp_out=$TMPDIR/bitedit.$pair.$n
        $GENERATOR \
            --seed 42 \
            --task $OP \
            --min-n-bits $old_n \
            --max-n-bits $n \
            --n-examples $SIZE \
            --zero-char $zero_char \
            --one-char $one_char \
            > $tmp_out

        awk '{if (NR <= 1000) print $0;}' $tmp_out > $TMPDIR/$pair.$n.valid
        awk '{if (NR > 1000 && NR <= 2000) print $0;}' $tmp_out > $TMPDIR/$pair.$n.test
        awk '{if (NR > 2000) print $0;}' $tmp_out > $TMPDIR/$pair.$n.train

        for t in train valid test; do
            f_in=$TMPDIR/$pair.$n.$t
            f_out=$OUTDIR/$pair.$n.$t
            cut -f2 $f_in > $f_out.x
            cut -f3 $f_in > $f_out.y
        done
    done
done

for t in train valid test; do
    for n in 15 20 25 30 35 40; do
        echo "Creating 'all.$n' dataset..." >&2
        cat $TMPDIR/*.$n.$t | shuf > $TMPDIR/all.$n.$t
        cut -f2 $TMPDIR/all.$n.$t > $OUTDIR/all.$n.$t.x
        cut -f3 $TMPDIR/all.$n.$t > $OUTDIR/all.$n.$t.y
    done
done

rm -r $TMPDIR
