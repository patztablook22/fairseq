#!/bin/bash

dir=$1
suf=$2

echo BLEU,COUNT
for d in $dir/*$suf; do
    #cat $d/iwslt17_tst14.best.bleu.txt

    for h in 0 1 2 3 4 5 6 7; do
        echo `cat $d/iwslt17_tst14.sel_$h.best.bleu.txt`,`./parse_logs.py --logfile $d/logs/tformer_mod.o* | grep ^VALID | sed "s/^.*$h-\([[:alnum:]]*\).*$/\1/" | tr "\n" "+" | sed 's/+[^0-9]*$/\n/' | bc`

        #count=`./parse_logs.py --logfile $d/logs/tformer_mod.o* | grep ^VALID | sed "s/^.*$h-\([[:alnum:]]*\).*$/\1/" | tr "\n" "+" | sed 's/+$/\n/'| bc 2> /dev/null`
        #bleu=`cat $d/iwslt17_tst14.sel_$h.best.bleu.txt`
        #echo "${bleu} ${count}"
    done

    #./parse_logs.py --logfile $d/logs/tformer_mod.o* --print-head-counts
done
