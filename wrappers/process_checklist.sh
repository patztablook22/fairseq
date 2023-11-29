#!/bin/bash
set -e

JOB_PRIORITY=10
SLURM_CONSTRAINTS=
EXCLUDE_NODES="dll-10gpu1,dll-10gpu2,dll-10gpu3"
SLURM_SUBMIT="/home/varis/scripts/slurm-submit.sh"

EXP_DIR=
TRANSLATION_OPT=""
BEAM_SIZE=1
LENPEN=0.6

SRC=x
TGT=y

USE_ORACLE=1
MODULE_MASK=
THRESHOLD=

CURRENT_TASK=
TASKS="id push pop shift unshift reverse"
LENGTHS=

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
    --use-oracle)
        USE_ORACLE=0
    ;;
    --module-mask)
        MODULE_MASK="$2"
        shift
    ;;
    --threshold)
        THRESHOLD="$2"
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

    hyps_file=$_file
    [[ -n "$MODULE_MASK" ]] && hyps_file="${hyps_file}.mask-$MODULE_MASK"

    # extract hypotheses and compute BLEU againts references
    grep '^H' $RESULTS_DIR/$hyps_file.txt \
        | sed 's/^H\-//' \
        | sort -n -k 1 \
        | cut -f3 \
        > $RESULTS_DIR/$hyps_file.hyps.txt
    msg "Evaluating $hyps_file.hyps.txt..."
    paste $RESULTS_DIR/$hyps_file.hyps.txt $EVAL_DIR/${_file}.$TGT | my_scripts/compare_sequences.py > $RESULTS_DIR/$hyps_file.eval_out
}

function translate {
    # The function takes two global variables (modifiers) for varying modes of translation:
    _file=$1
    _sys=$2

    outfile=$RESULTS_DIR/${_file}

    #cmd="source $VIRTUALENV && export CUDA_LAUNCH_BLOCKING=1"
    cmd="cat $EVAL_DIR/${_file}.$SRC"
    cmd="$cmd | wrappers/translate_wrapper_interactive.sh $_sys '_$CURRENT_TASK' $outfile '$TRANSLATION_OPT'"
    cmd="$cmd && mv $outfile.$CURRENT_TASK.txt $outfile.txt"

    [[ -e "$outfile.txt" ]] && exit 0

    $cmd
    #jid=`$SLURM_SUBMIT --jobname tr_eval --constraints "$SLURM_CONSTRAINTS" --exclude "$EXCLUDE_NODES" --logdir logs --gpus $GPUS --mem $GPUMEM --cores $CORES --priority $JOB_PRIORITY "$cmd"`
    #jid=`echo $jid | cut -d" " -f4`
    #echo $jid
}

function extract_attention {
    _file=$1
    _sys=$2

    infile="$EVAL_DIR/${_file}.x"
    outfile="$EVAL_DIR/$RESULTS_DIR/attention/${_file}"
    mkdir -p $RESULTS_DIR/attention

    #cmd="source $VIRTUALENV"
    cmd="my_scripts/extract_attention.py --input-file $infile --checkpoint $_sys/checkpoints/checkpoint_$CURRENT_TASK.pt --output-file $outfile"

    $cmd
    #jid=`qsubmit --jobname=tr_att --logdir=logs --gpus=$GPUS --gpumem=$GPUMEM --mem=$MEM --cores=$CORES --priority=$JOB_PRIORITY "$cmd"`
    #jid=`echo $jid | cut -d" " -f3`
    #echo $jid
}

function process_files {
    _dataset=$1
    _dir=$2

    for task in $TASKS; do 
        msg "Processing $task.$_dataset ..."

        translate $task.$_dataset $EXP_DIR
        #jid=`translate $task.$_dataset $EXP_DIR`
        #msg "Waiting for job $jid..."
        #while true; do
        #    [[ -z $jid ]] && break
        #    sleep 20
        #    #qstat | grep $jid > /dev/null || break
        #    squeue | grep " $jid " > /dev/null || break
        #done
        evaluate $task.$_dataset $EXP_DIR

        for len in $LENGTHS; do
            msg "Processing $task.$len.$_dataset ..."

                
            translate $task.$len.$_dataset $EXP_DIR
            #jid=`translate $task.$len.$_dataset $EXP_DIR`
            #msg "Waiting for job $jid..."
            #while true; do
            #    [[ -z $jid ]] && break
            #    sleep 20
            #    #qstat | grep $jid > /dev/null || break
            #    squeue | grep " $jid " > /dev/null || break
            #done
            evaluate $task.$len.$_dataset $EXP_DIR

            #mkdir -p $EVAL_DIR/$OUTPUT_DIR/attention
            #extract_attention $task.$len.$_dataset $EXP_DIR
        done
    done
}

RESULTS_DIR=$EXP_DIR/$CURRENT_TASK.bs-$BEAM_SIZE.lp-$LENPEN.eval
[[ $OVERWRITE -eq 0 ]] && [[ -d $RESULTS_DIR ]] && rm -r $RESULTS_DIR
[[ -d "$RESULTS_DIR" ]] || mkdir -p $RESULTS_DIR

process_files $EVAL_DATASET $EVAL_DIR

# TODO: include clustering eval
