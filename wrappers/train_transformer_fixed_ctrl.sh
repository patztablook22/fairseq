#!/bin/bash
set -e

JOB_PRIORITY=-90
GPUMEM="11g"

EXPDIR=
EVAL_DIR=
TASKS="id flip reverse flip-reverse"
VALID_TASKS=$TASKS

# General Architecture Details
EMB_SIZE=128
FFN_SIZE=$(expr 4 \* $EMB_SIZE)
ATT_HEADS=8
DEPTH=1
SHARED_DICT_OPT="--share-decoder-input-output-embed"

# Training Reset
RESET_OPTIMIZER_OPT=

# Training Details
RANDOM_SEED=42
EPOCHS=100
LABEL_SMOOTHING=0.1
MAX_TOKENS=4096
DROPOUT=0.3
LR="5e-4"
WARMUP=4000
CLIP_NORM=0.0
PATIENCE=0
KEEP_N_CHECKPOINTS=1
SAVE_EVERY_N_UPDATES=0

# Modular Details
CTRL_TYPE="attention"
CTRL_DEPTH=0
CTRL_DIM=$EMB_SIZE
CTRL_DROP=0.0
#CTRL_SAMPLING_OPT=""
#CTRL_AVG_TOKENS_OPT=""

# We do not need ctrl training hyperparams since we use fixed_module_mask for each training task
# Gumbel Temperature
#CTRL_MIN_TEMP=0.0625
#CTRL_MAX_TEMP=1.
#CTRL_ANNEAL_TYPE="exponential"
#CTRL_ANNEAL_RATE="1e-6"

# Regularizer
#CTRL_MODULE_COVERAGE=0.5
#CTRL_REGULARIZER_WEIGHT=1.0

FREEZE_PARAMS=
#FREEZE_PARAMS='ctrl_,final_layer,fc1,fc2,embed_'


HELP=1
while [[ $# -gt 0 ]]; do
key="$1"
case $key in
    -e|--expdir)
        EXPDIR="$2"
        shift
    ;;
    --eval-dir)
        EVAL_DIR="$2"
        shift
    ;;
    -r|--random-seed)
        RANDOM_SEED="$2"
        shift
    ;;
    -t|--tasks)
        TASKS="$2"
        shift
    ;;
    --valid-tasks)
        VALID_TASKS="$2"
        shift
    ;;
    --epochs)
        EPOCHS="$2"
        shift
    ;;
    --emb-size)
        EMB_SIZE="$2"
        shift
    ;;
    --ffn-size)
        FFN_SIZE="$2"
        shift
    ;;
    --att-heads)
        ATT_HEADS="$2"
        shift
    ;;
    --depth)
        DEPTH="$2"
        shift
    ;;
    --lr|--learning-rate)
        LR="$2"
        shift
    ;;
    --warmup)
        WARMUP="$2"
        shift
    ;;
    --max-tokens)
        MAX_TOKENS="$2"
        shift
    ;;
    --label-smoothing)
        LABEL_SMOOTHING="$2"
        shift
    ;;
    --clip-norm)
        CLIP_NORM="$2"
        shift
    ;;
    --patience)
        PATIENCE="$2"
        shift
    ;;
    --keep-n)
        KEEP_N_CHECKPOINTS="$2"
        shift
    ;;
    --ctrl-dim)
        CTRL_DIM="$2"
        shift
    ;;
    --ctrl-depth)
        CTRL_DEPTH="$2"
        shift
    ;;
    --ctrl-dropout)
        CTRL_DROP="$2"
        shift
    ;;
    --ctrl-type)
        CTRL_TYPE="$2"
        shift
    ;;
    #--ctrl-hard-samples)
    #    CTRL_SAMPLING_OPT="--module-ctrl-hard-samples"
    #;;
    #--ctrl-avg-tokens)
    #    CTRL_AVG_TOKENS_OPT="--module-ctrl-avg-tokens"
    #;;
    #--ctrl-min-temp)
    #    CTRL_MIN_TEMP="$2"
    #    shift
    #;;
    #--ctrl-max-temp)
    #    CTRL_MAX_TEMP="$2"
    #    shift
    #;;
    #--ctrl-anneal-type)
    #    CTRL_ANNEAL_TYPE="$2"
    #    shift
    #;;
    #--ctrl-anneal-rate)
    #    CTRL_ANNEAL_RATE="$2"
    #    shift
    #;;
    #--ctrl-coverage)
    #    CTRL_MODULE_COVERAGE="$2"
    #    shift
    #;;
    #--ctrl-regularizer-weight)
    #    CTRL_REGULARIZER_WEIGHT="$2"
    #    shift
    #;;
    --freeze-params)
        FREEZE_PARAMS="$2"
        shift
    ;;
    --save-every-n)
        SAVE_EVERY_N_UPDATES="$2"
        shift
    ;;
    --reset-optimizer)
        RESET_OPTIMIZER_OPT="--reset-optimizer"
    ;;
    --shared-dict)
        SHARED_DICT_OPT="--share-all-embeddings"
    ;;
    -h|--help)
        HELP=0
        shift
    ;;
    *)
        echo Unknown option '"'$key'"' >&2
        exit 1
    ;;
esac
shift
done

[[ -d "$EXPDIR" ]] || exit 1
[[ -d "$EVAL_DIR" ]] || exit 1
# TODO print help

MODEL_DIR="$EXPDIR/transformer_modular"
MODEL_DIR="$MODEL_DIR.seed-$RANDOM_SEED"
MODEL_DIR="$MODEL_DIR.warmup-$WARMUP"
MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
#MODEL_DIR="$MODEL_DIR.ffn-size-$FFN_SIZE"
MODEL_DIR="$MODEL_DIR.att-heads-$ATT_HEADS"
MODEL_DIR="$MODEL_DIR.depth-$DEPTH"
MODEL_DIR="$MODEL_DIR.lr-$LR"
#MODEL_DIR="$MODEL_DIR.ctrl-depth-$CTRL_DEPTH"
#MODEL_DIR="$MODEL_DIR.max-tokens-$MAX_TOKENS"
[[ -z "$RESET_OPTIMIZER_OPT" ]] || MODEL_DIR="$MODEL_DIR.reset-optim"
#[[ -z "$CTRL_AVG_TOKENS_OPT" ]] || MODEL_DIR="$MODEL_DIR.mod-avg-tokens"
#[[ -z "$CTRL_SAMPLING_OPT" ]] || MODEL_DIR="$MODEL_DIR.mod-hard-samples"
#MODEL_DIR="$MODEL_DIR.anneal-type-$CTRL_ANNEAL_TYPE"
#MODEL_DIR="$MODEL_DIR.coverage-$CTRL_MODULE_COVERAGE"
#MODEL_DIR="$MODEL_DIR.max-temp-$CTRL_MAX_TEMP"
#MODEL_DIR="$MODEL_DIR.reg-weight-$CTRL_REGULARIZER_WEIGHT"
MODEL_DIR="$MODEL_DIR.epochs-$EPOCHS"


[[ -d $MODEL_DIR ]] && rm -r $MODEL_DIR
mkdir $MODEL_DIR
echo $TASKS | sed 's/ /->/g' > $MODEL_DIR/TASKS
echo $FREEZE_PARAMS > $MODEL_DIR/FREEZE_PARAMS

ckpt_opt=
epochs=$EPOCHS
valid_sets=`echo $VALID_TASKS | sed 's/ /.15.valid,/g;s/$/.15.valid/'`
for current_task in $TASKS; do
    echo Training $current_task...

    MODULE_MASK_OPT=
    [[ -e "$EXPDIR/data/${current_task}.mask" ]] && MODULE_MASK_OPT="--module-ctrl-fixed-mask $(cat $EXPDIR/data/${current_task}.mask)"
    PARAM_FREEZE_OPT=
    [[ -e "$MODEL_DIR/checkpoints/checkpoint_last.pt" ]] && PARAM_FREEZE_OPT="--parameter-freeze-substr '$FREEZE_PARAMS'"

    jid=$(qsubmit \
        --queue="gpu-troja.q" \
        --logdir=$MODEL_DIR/logs \
        --jobname=tformer_bit_mod \
        --mem=25g \
        --cores=4 \
        --gpumem=$GPUMEM \
        --priority=$JOB_PRIORITY \
        --gpus=1 "source $HOME/python-virtualenv/fairseq-env/bin/activate && \
            export CUDA_LAUNCH_BLOCKING=1 && \
            python train.py \
            $EXPDIR/data \
            -s x \
            -t y \
            --seed $RANDOM_SEED \
            --task translation_modular \
            --arch transformer_modular \
            $SHARED_DICT_OPT \
            --share-decoder-input-output-embed \
            --train-subset ${current_task}.15.train \
            --valid-subset $valid_sets \
            $ckpt_opt \
            $RESET_OPTIMIZER_OPT \
            --optimizer adam \
            --adam-betas '(0.9, 0.98)' \
            --clip-norm $CLIP_NORM \
            --patience $PATIENCE \
            --max-epoch $epochs \
            --lr $LR \
            --lr-scheduler inverse_sqrt \
            --warmup-updates $WARMUP \
            --dropout $DROPOUT \
            --weight-decay 0.0001 \
            --criterion label_smoothed_cross_entropy_modular \
            --label-smoothing $LABEL_SMOOTHING \
            --max-tokens $MAX_TOKENS \
            --eval-acc \
            --eval-ter \
            --eval-bleu \
            --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
            --eval-bleu-detok moses \
            --eval-bleu-remove-bpe \
            --eval-bleu-print-samples \
            --best-checkpoint-metric ter \
            --tensorboard-logdir $MODEL_DIR \
            --save-dir $MODEL_DIR/checkpoints \
            --keep-last-epochs $KEEP_N_CHECKPOINTS \
            --encoder-attention-heads $ATT_HEADS \
            --decoder-attention-heads $ATT_HEADS \
            --encoder-layers $DEPTH \
            --decoder-layers $DEPTH \
            --encoder-embed-dim $EMB_SIZE \
            --encoder-ffn-embed-dim $FFN_SIZE \
            --module-ctrl-hidden-depth $CTRL_DEPTH \
            --module-ctrl-hidden-dim $CTRL_DIM \
            --module-ctrl-word-dropout $CTRL_DROP \
            --module-ctrl-avg-tokens \
            --module-ctrl-hard-samples \
            --module-coverage-regularizer-weight 0 \
            $PARAM_FREEZE_OPT \
            $MODULE_MASK_OPT \
            --save-interval-updates $SAVE_EVERY_N_UPDATES")

    jid=`echo $jid | cut -d" " -f3`
    echo Waiting for $jid...
    while true; do
        sleep 15
        qstat | grep $jid > /dev/null || break
    done
    cp $MODEL_DIR/checkpoints/checkpoint_last.pt \
        $MODEL_DIR/checkpoints/checkpoint_$current_task.pt
    ckpt_opt="--restore-file $MODEL_DIR/checkpoints/checkpoint_$current_task.pt"

    MODULE_MASK_OPT=
    [[ -e "$EXPDIR/data/${current_task}.mask" ]] && MODULE_MASK_OPT="--module-mask $(cat $EXPDIR/data/${current_task}.mask)"

    echo "Finished $current_task. Evaluating..."

    for eval_task in $TASKS; do
        EVAL_MASK_OPT=$MODULE_MASK_OPT
        [[ -e "$EXPDIR/data/${eval_task}.mask" ]] && EVAL_MASK_OPT="--module-mask $(cat $EXPDIR/data/${eval_task}.mask)"

        bash process_checklist.sh \
            -t $current_task \
            --expdir $MODEL_DIR \
            --tasks "$VALID_TASKS $current_task" \
            --eval-dir $EVAL_DIR \
            $EVAL_MASK_OPT \
            --translation-options "--print-module-mask" | tee -a $MODEL_DIR/logs/$current_task.eval.log &
    done
    epochs=`expr $epochs + $EPOCHS`
done
