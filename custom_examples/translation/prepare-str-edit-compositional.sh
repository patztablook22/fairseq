#!/usr/bin/env bash
set -e
WD=`pwd`

GENERATOR="../../my_scripts/create_bitedit_examples.py"
SIZE=${1:-30000}
OPS="id reverse flip flip-reverse"
ZERO_SHOT_TASK="flip-reverse"

OUTDIR=$WD/str-edit.compo.$SIZE
TMPDIR=$OUTDIR/tmp.d

#prep=bitedit.compo.$SIZE
#tmp=$prep/tmp
#orig=$prep/orig

mkdir -p $OUTDIR $TMPDIR

for op in $OPS; do
    zero_char="a"
    one_char="b"
    pad_to_length=70
    end_char="c"
    pad_char="d"

    old_n=1
    for n in 20 30 40 50 60; do
        tmp_out=$TMPDIR/$op.$n
        echo "Creating '$tmp_out' dataset..." >&2

        $GENERATOR \
            --seed 42 \
            --task $op \
            --min-n-bits $old_n \
            --max-n-bits $n \
            --n-examples $SIZE \
            --zero-char $zero_char \
            --one-char $one_char \
            --pad-to-length $pad_to_length \
            --end-char $end_char \
            --pad-char $pad_char \
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

        old_n=$(expr $n + 1)
    done
done

for t in train valid test; do
    for n in 20 30 40 50 60; do
        f_in=$TMPDIR/all.$n.$t
        f_out=$OUTDIR/all.$n.$t
        echo "Creating '$f_out' dataset..." >&2

        cat $TMPDIR/*.$n.$t \
            | grep -v $ZERO_SHOT_TASK \
            | shuf \
            > $f_in
        cut -f1,2 $f_in | sed 's/\t/ | /' > $f_out.x
        cut -f3 $f_in > $f_out.y
    done
done

for f in $OUTDIR/*x; do
    sed -i 's/^id |/0 0 |/' $f
    sed -i 's/^flip-reverse |/1 1 |/' $f
    sed -i 's/^flip |/1 0 |/' $f
    sed -i 's/^reverse |/0 1 |/' $f
done

rm -r $TMPDIR
