#!/usr/bin/env bash
set -e
WD=`pwd`

GENERATOR="../../my_scripts/create_bitedit_examples.py"
SIZE=${1:-30000}
OPS=${2:-"id push pop shift unshift remove reverse duplicate flip"}

OUTDIR=$WD/str-edit.multitask.`echo $OPS | tr " " "-"`.$SIZE
TMPDIR=$OUTDIR/tmp.d

ZERO_CHAR="a"
ONE_CHAR="b"

#prep=bitedit.$SIZE
#tmp=$prep/tmp
#orig=$prep/orig

mkdir -p $OUTDIR $TMPDIR

old_n=1
for op in $OPS; do
    for n in 15 20 25 30 35 40; do
        echo "Creating '$op.$n' dataset..." >&2
        tmp_out="$TMPDIR/$op.$n"
        $GENERATOR \
            --seed 42 \
            --task $op \
            --min-n-bits $old_n \
            --max-n-bits $n \
            --n-examples $SIZE \
            --zero-char $ZERO_CHAR \
            --one-char $ONE_CHAR \
            > $tmp_out

        awk '{if (NR <= 1000) print $0;}' $tmp_out > $tmp_out.valid
        awk '{if (NR > 1000 && NR <= 2000) print $0;}' $tmp_out > $tmp_out.test
        awk '{if (NR > 2000) print $0;}' $tmp_out > $tmp_out.train

        for t in train valid test; do
            f_in=$TMPDIR/$op.$n.$t
            f_out=$OUTDIR/$op.$n.$t
            cut -f1,2 $f_in | sed 's/\t/ | /' > $f_out.x
            cut -f3 $f_in > $f_out.y
        done
    done
done

for t in train valid test; do
    for n in 15 20 25 30 35 40; do
        echo "Creating 'all.$n' dataset..." >&2
        cat $TMPDIR/*.$n.$t | shuf > $TMPDIR/all.$n.$t
        cut -f1,2 $TMPDIR/all.$n.$t | sed 's/\t/ | /' > $OUTDIR/all.$n.$t.x
        cut -f3 $TMPDIR/all.$n.$t > $OUTDIR/all.$n.$t.y
    done
done

rm -r $TMPDIR
