#!/usr/bin/env bash
set -e
WD=`pwd`

GENERATOR="../../my_scripts/create_bitedit_examples.py"
SIZE=${1:-5000}
OPS=${2:-"id push pop shift unshift remove reverse duplicate flip"}

prep=bitedit.$SIZE
tmp=$prep/tmp
orig=$prep/orig

mkdir -p $orig $tmp $prep

old_n=1
for op in $OPS; do
    for n in 15 20 25 30 35 40; do
        echo "Creating '$op.$n' dataset..." >&2
        $GENERATOR \
            --seed 42 \
            --task $op \
            --min-n-bits $old_n \
            --max-n-bits $n \
            --n-examples $SIZE > $orig/bitedit.$op.$n

        awk '{if (NR <= 1000) print $0;}' $orig/bitedit.$op.$n > $orig/$op.$n.valid
        awk '{if (NR > 1000 && NR <= 2000) print $0;}' $orig/bitedit.$op.$n > $orig/$op.$n.test
        awk '{if (NR > 2000) print $0;}' $orig/bitedit.$op.$n > $orig/$op.$n.train

        for t in train valid test; do
            cut -f1,2 $orig/$op.$n.$t | sed 's/\t/ | /' > $prep/$op.$n.$t.x
            cut -f3 $orig/$op.$n.$t > $prep/$op.$n.$t.y
        done
    done
done

for t in train valid test; do
    for n in 15 20 25 30 35 40; do
        echo "Creating '$op.$n' dataset..." >&2
        cat $orig/*.$n.$t | shuf > $orig/all.$n.$t
        cut -f1,2 $orig/all.$n.$t | sed 's/\t/ | /' > $prep/all.$n.$t.x
        cut -f3 $orig/all.$n.$t > $prep/all.$n.$t.y
    done
done
