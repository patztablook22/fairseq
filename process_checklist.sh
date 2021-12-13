#!/bin/bash
set -e

JOB_PRIORITY=-65

EXP_DIR=
TRANSLATION_OPT=""

SRC=en
TGT=cs

USE_ORACLE=1
SELECTION=

CURRENT_TASK=
TASKS="newstest"
LENGTHS="10 20 30 40 50 60 70 80 90 100"

EVAL_DATASET="test"
EVAL_DIR="custom_examples/translation/wmt20_encs"

HELP=1
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --expdir)
        EXP_DIR="$2"
        shift
    ;;
    --eval-prefix)
        EVAL_DATASET="$2"
        shift
    ;;
    --eval-dir)
        EVAL_DIR="$2"
        shift
    ;;
    --src)
        SRC="$2"
        shift
    ;;
    --tgt)
        TGT="$2"
        shift
    ;;
    -t|--current-task)
        CURRENT_TASK="$2"
        shift
    ;;
    --tasks)
        TASKS="$2"
        shift
    ;;
    --translation-options)
        TRANSLATION_OPT="$2"
        shift
    ;;
    -h|--help)
        HELP=0
    ;;
    --use-oracle)
        USE_ORACLE=0
    ;;
    --selection)
        SELECTION="$2"
        shift
    ;;
    -h|--help)
        HELP=0
    ;;
    *)
        echo Unknown option '"'$key'"' >&2
        exit 1
    ;;
esac
shift
done

# CONSTANTS
VIRTUALENV="/home/varis/python-virtualenv/fairseq-env/bin/activate"
CORES=4
MEM="10g"
GPUMEM="11g"
GPUS=1

DETOKENIZER=mosesdecoder/scripts/tokenizer/detokenizer.perl

TRANSLATION_OPT="-s $SRC -t $TGT --bpe subword_nmt --bpe-codes $EVAL_DIR/bpecodes"

# TODO print help


function msg {
    echo "`date '+%Y-%m-%d %H:%M:%S'`  |  $@" >&2
}

function evaluate {
    _file=$1
    _sys=$2

    grep '^H' $_sys/$CURRENT_TASK.eval/$_file.txt | \
        sed 's/^H\-//' | \
        sort -n -k 1 | \
        cut -f3 | \
        perl $DETOKENIZER -l $TGT | \
        sed "s/ - /-/g" > $_sys/$CURRENT_TASK.eval/$_file.hyps.detok.txt
    msg "Evaluating $_file.hyps.detok.txt..."
    sacrebleu --input $_sys/$CURRENT_TASK.eval/$_file.hyps.detok.txt $EVAL_DIR/${_file}.$TGT > $_sys/$CURRENT_TASK.eval/${_file}.eval_out
}

function translate {
    # The function takes two global variables (modifiers) for varying modes of translation:
    # (USE_ORACLE and SELECTION)
    _file=$1
    _sys=$2

    cmd="source $VIRTUALENV"
    if [[ $USE_ORACLE -eq 0 ]]; then
        oracle_input="$EVAL_DIR/${_file}.$SRC $EVAL_DIR/${_file}.$TGT"
        outfile=$CURRENT_TASK.eval/${_file}.oracle
        cmd="$cmd && wrappers/translate_modular_oracle_interactive.sh $_sys <(paste $oracle_input) '_$CURRENT_TASK' $outfile '$TRANSLATION_OPT --print-selection --print-attn-confidence'"
    elif [[ -n "$SELECTION" ]]; then
        oracle_input="$EVAL_DIR/${_file}.$SRC $EVAL_DIR/${_file}.$TGT"
        outfile=$CURRENT_TASK.eval/${_file}.sel_$SELECTION
        cmd="$cmd && wrappers/translate_modular_oracle_interactive.sh $_sys <(paste $oracle_input) '_$CURRENT_TASK' $outfile '$TRANSLATION_OPT --print-selection --print-attn-confidence --fixed-encoder-selection ($SELECTION) --fixed-decoder-selection ($SELECTION)'"
    else
        outfile=$CURRENT_TASK.eval/${_file}
        cmd="$cmd && cat $EVAL_DIR/${_file}.$SRC"
        cmd="$cmd | wrappers/translate_wrapper_interactive.sh $_sys '_$CURRENT_TASK' $outfile '$TRANSLATION_OPT'"
    fi
    cmd="$cmd && mv $_sys/$outfile.$CURRENT_TASK.txt $_sys/$outfile.txt"
    [[ -e "$_sys/$outfile.txt" ]] && exit 0

    jid=`qsubmit --jobname=tr_len_eval --logdir=logs --gpus=$GPUS --gpumem=$GPUMEM --mem=$MEM --cores=$CORES --priority=$JOB_PRIORITY "$cmd"`
    jid=`echo $jid | cut -d" " -f3`
    echo $jid
}

function process_files {
    _dataset=$1
    _dir=$2

    for len in $LENGTHS; do
        for task in $TASKS; do 
            msg "Processing $task.$len.$_dataset ..."

            jid=`translate $task.$len.$_dataset $EXP_DIR`
            msg "Waiting for job $jid..."
            while true; do
                sleep 20
                qstat | grep $jid > /dev/null || break
            done
            evaluate $task.$len.$_dataset $EXP_DIR
        done
    done
}

[[ -d "$EXP_DIR/$CURRENT_TASK.eval" ]] || mkdir $EXP_DIR/$CURRENT_TASK.eval
process_files $EVAL_DATASET $EVAL_DIR

# TODO: include clustering eval
