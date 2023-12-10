#!/bin/bash
set -eou pipefail
# CONSTANTS
VIRTUALENV="/home/varis/python-virtualenv/fairseq-env/bin/activate"
MEM="10g"
GPUMEM="11g"
GPUS=1
SERIES_AVG=my_scripts/average_dynamic_series.py

SRC=x
TGT=y

EXP_DIR=
TRANSLATION_OPT=""
BEAM_SIZE=1
LENPEN=0.6

TRANSLATION_OPT="-s $SRC -t $TGT --beam $BEAM_SIZE --lenpen $LENPEN $TRANSLATION_OPT"

JOB_PRIORITY=10
SLURM_CONSTRAINTS=
PARTITION="gpu-ms,gpu-troja"

CURRENT_TASK=
TASKS="reverse.30 all.30"

EVAL_DATASET="test"
EVAL_DIR="custom_examples/translation/str-edit.compo.30000"

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

# TODO print help


function msg {
    echo "`date '+%Y-%m-%d %H:%M:%S'`  |  $@" >&2
}

function evaluate {
    _file=$1
    _sys=$2

    hyps_file=$_file

    # extract hypotheses and compute BLEU againts references
    grep '^H' $RESULTS_DIR/$hyps_file.txt \
        | sed 's/^H\-//' \
        | sort -n -k 1 \
        | cut -f3 \
        > $RESULTS_DIR/$hyps_file.hyps.txt
    msg "Evaluating $hyps_file.hyps.txt..."
    paste $RESULTS_DIR/$hyps_file.hyps.txt $EVAL_DIR/${_file}.$TGT | my_scripts/compare_sequences.py > $RESULTS_DIR/$hyps_file.eval_out
}

function dump {
    # The function takes two global variables (modifiers) for varying modes of translation:
    _file=$1
    _sys=$2

    outfile=$RESULTS_DIR/${_file}

    in_x=$EVAL_DIR/${_file}.$SRC
    in_y=$EVAL_DIR/${_file}.$TGT
    out_x=$RESULTS_DIR/$CURRENT_TASK.x
    out_y=$RESULTS_DIR/$CURRENT_TASK.y
    out_hyps=$RESULTS_DIR/$CURRENT_TASK.hyps

    tempfile="$SCRATCH/temp_${_file}_$(date +%s)"

    #cmd="cat $in_x | wrappers/translate_wrapper_interactive.sh $_sys '_$CURRENT_TASK' $tempfile '$TRANSLATION_OPT'"
    #bash -c "$cmd"

    wrappers/translate_wrapper_interactive.sh \
        $in_x \
        "$EXP_DIR" "_$CURRENT_TASK" "$tempfile" $TRANSLATION_OPT

    grep '^H' $tempfile.$CURRENT_TASK.txt \
        | sed 's/^H\-//' \
        | sort -n -k 1 \
        | cut -f3 \
        >> $out_hyps

    cat $in_x >> $out_x
    cat $in_y >> $out_y
}

function translate {
    # The function takes two global variables (modifiers) for varying modes of translation:
    _file=$1
    _sys=$2

    outfile=$RESULTS_DIR/${_file}

    cmd="source $VIRTUALENV && export CUDA_LAUNCH_BLOCKING=1"
    cmd="$cmd && cat $EVAL_DIR/${_file}.$SRC"
    cmd="$cmd | wrappers/translate_wrapper_interactive.sh $_sys '_$CURRENT_TASK' $outfile '$TRANSLATION_OPT'"
    cmd="$cmd && mv $outfile.$CURRENT_TASK.txt $outfile.txt"

    [[ -e "$outfile.txt" ]] && exit 0
	mkdir -p logs
    srun \
        -J "tr_eval" \
        -o "logs/tr_eval.o%j" \
        -e "logs/tr_eval.o%j" \
        -p "$PARTITION" \
        -C "$SLURM_CONSTRAINTS" \
        --mem $GPUMEM \
        --gpus $GPUS \
        --priority $JOB_PRIORITY \
        bash -c "$cmd"
}

function extract_attention {
	# pass
	sleep 1
}

function process_files {
    _dataset=$1
    _dir=$2

    for task in $TASKS; do 
        msg "Processing $task.$_dataset ..."
        dump $task.$_dataset $EXP_DIR $RESULTS_DIR
    done
}

RESULTS_DIR=$EXP_DIR/$CURRENT_TASK.bs-$BEAM_SIZE.lp-$LENPEN.eval
[[ $OVERWRITE -eq 0 ]] && [[ -d $RESULTS_DIR ]] && rm -r $RESULTS_DIR
[[ -d "$RESULTS_DIR" ]] || mkdir -p $RESULTS_DIR

process_files $EVAL_DATASET $EVAL_DIR


PROJECT_DIR="$(dirname $0)/.."
mkdir $PROJECT_DIR/results

echo "cp -r $RESULTS_DIR $PROJECT_DIR/results ..."
cp -r $RESULTS_DIR $PROJECT_DIR/results

# TODO: include clustering eval
