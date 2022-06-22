#!/bin/bash
set -e

JOB_PRIORITY=-85

EXP_DIR=
TRANSLATION_OPT=""
BEAM_SIZE=1
LENPEN=0.6

SRC=x
TGT=y

CURRENT_TASK=
TASKS="id push pop shift unshift reverse"
LENGTHS="15 20 25"

EVAL_DATASET="test"
EVAL_DIR="custom_examples/translation/bitedit.30000"

OVERWRITE=1

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
    -bs|--beam-size)
        BEAM_SIZE="$2"
        shift
    ;;
    --lenpen)
        LENPEN="$2"
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
    --task-lengths)
        LENGTHS="$2"
        shift
    ;;
    --translation-options)
        TRANSLATION_OPT="$2"
        shift
    ;;
    --overwrite)
        OVERWRITE=0
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

SERIES_AVG=my_scripts/average_dynamic_series.py

TRANSLATION_OPT="-s $SRC -t $TGT --beam $BEAM_SIZE --lenpen $LENPEN $TRANSLATION_OPT"

# TODO print help


function msg {
    echo "`date '+%Y-%m-%d %H:%M:%S'`  |  $@" >&2
}

function evaluate {
    _file=$1
    _sys=$2

    # extract hypotheses and compute BLEU against references
    grep '^H' $RESULTS_DIR/$_file.txt \
        | sed 's/^H\-//' \
        | sort -n -k 1 \
        | cut -f3 \
        > $RESULTS_DIR/$_file.hyps.txt
    msg "Evaluating $_file.hyps.txt..."
    paste $RESULTS_DIR/$_file.hyps.txt $EVAL_DIR/${_file}.y | my_scripts/compare_sequences.py > $RESULTS_DIR/$_file.eval_out

    # extract and agregate EOS emission probs
    #grep 'eos-lprobs' $RESULTS_DIR/$_file.txt \
    #    | cut -f2 \
    #    | $SERIES_AVG \
    #    > $RESULTS_DIR/${_file}.eos_lprobs
}

function translate {
    _file=$1
    _sys=$2

    outfile=$RESULTS_DIR/${_file}

    cmd="source $VIRTUALENV"
    cmd="$cmd && cat $EVAL_DIR/${_file}.$SRC"
    cmd="$cmd | wrappers/translate_wrapper_interactive.sh $_sys '_$CURRENT_TASK' $outfile '$TRANSLATION_OPT'"
    cmd="$cmd && mv $outfile.$CURRENT_TASK.txt $outfile.txt"

    [[ -e "$outfile.txt" ]] && exit 0

    jid=`qsubmit --jobname=tr_eval --logdir=logs --gpus=$GPUS --gpumem=$GPUMEM --mem=$MEM --cores=$CORES --priority=$JOB_PRIORITY "$cmd"`
    jid=`echo $jid | cut -d" " -f3`
    echo $jid
}

function process_files {
    _dataset=$1
    _dir=$2

    my_scripts/print_model_variables.py \
        --checkpoint $EXP_DIR/checkpoints/checkpoint_$CURRENT_TASK.pt \
    | grep fisher | cut -f2 | ~/scripts/min_max_avg.py > $RESULTS_DIR/fisher.min

    my_scripts/print_model_variables.py \
        --checkpoint $EXP_DIR/checkpoints/checkpoint_$CURRENT_TASK.pt \
    | grep fisher | cut -f3 | ~/scripts/min_max_avg.py > $RESULTS_DIR/fisher.max

    for task in $TASKS; do
        for len in $LENGTHS; do
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

RESULTS_DIR=$EXP_DIR/$CURRENT_TASK.bs-$BEAM_SIZE.lp-$LENPEN.eval
[[ $OVERWRITE -eq 0 ]] && [[ -d $RESULTS_DIR ]] && rm -r $RESULTS_DIR
[[ -d "$RESULTS_DIR" ]] || mkdir $RESULTS_DIR

process_files $EVAL_DATASET $EVAL_DIR
