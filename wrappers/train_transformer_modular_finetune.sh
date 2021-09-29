#!/bin/bash
set -e

JOB_PRIORITY=-90
GPUMEM="11g"

EXPDIR=
EVAL_DIR=
EXP_SUFFIX=
TASKS="all id flip reverse flip-reverse all.tuning"
VALID_TASKS=$TASKS

# General Architecture Details
EMB_SIZE=128
FFN_SIZE=$(expr 4 \* $EMB_SIZE)
ATT_HEADS=8
DEPTH=1
SHARED_DICT_OPT=

# Training Reset
RESET_OPTIMIZER_OPT=
INIT_CKPT=

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

# Validation - Beam Search Details
VALID_BEAM_SIZE=5
VALID_MAX_LEN_A=1.2
VALID_MAX_LEN_B=10

# Modular Details
CTRL_TYPE="attention"
CTRL_DEPTH=0
CTRL_DIM=$EMB_SIZE
CTRL_DROP=0.0
CTRL_SAMPLING_OPT=""
CTRL_AVG_TOKENS_OPT=""
CTRL_OUTPUT_BIAS_OPT=""

# Gumbel Temperature
CTRL_MIN_TEMP=0.0625
CTRL_MAX_TEMP=1.
CTRL_ANNEAL_TYPE="exponential"
CTRL_ANNEAL_RATE="1e-6"
CTRL_COSINE_DECAY=0.95
CTRL_COSINE_RESET=1.

# Regularizer
CTRL_KL_DIV_RATIO=0.5
CTRL_KL_DIV_WEIGHT=0.
CTRL_BUDGET_RATIO=0.5
CTRL_BUDGET_WEIGHT=0.

#FREEZE_PARAMS='ctrl_,final_layer,fc1,fc2,embed_'
# + 'att' during final (controller-only) fine-tune

OVERWRITE=1

HELP=1
while [[ $# -gt 0 ]]; do
key="$1"
case $key in
    -e|--expdir)
        EXPDIR="$2"
        shift
    ;;
    --exp-suffix)
        EXP_SUFFIX="$2"
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
    --ctrl-hard-samples)
        CTRL_SAMPLING_OPT="--module-ctrl-hard-samples"
    ;;
    --ctrl-avg-tokens)
        CTRL_AVG_TOKENS_OPT="--module-ctrl-avg-tokens"
    ;;
    --ctrl-add-bias)
        CTRL_OUTPUT_BIAS_OPT="--module-ctrl-add-output-bias"
    ;;
    --ctrl-min-temp)
        CTRL_MIN_TEMP="$2"
        shift
    ;;
    --ctrl-max-temp)
        CTRL_MAX_TEMP="$2"
        shift
    ;;
    --ctrl-anneal-type)
        CTRL_ANNEAL_TYPE="$2"
        shift
    ;;
    --ctrl-anneal-rate)
        CTRL_ANNEAL_RATE="$2"
        shift
    ;;
    --ctrl-cosine-decay)
        CTRL_COSINE_DECAY="$2"
        shift
    ;;
    --ctrl-cosine-reset)
        CTRL_COSINE_RESET="$2"
        shift
    ;;
    --ctrl-kl-div-ratio)
        CTRL_KL_DIV_RATIO="$2"
        shift
    ;;
    --ctrl-kl-div-weight)
        CTRL_KL_DIV_WEIGHT="$2"
        shift
    ;;
    --ctrl-budget-ratio)
        CTRL_BUDGET_RATIO="$2"
        shift
    ;;
    --ctrl-budget-weight)
        CTRL_BUDGET_WEIGHT="$2"
        shift
    ;;
    --valid-beam-size)
        VALID_BEAM_SIZE="$2"
        shift
    ;;
    --valid-max-len-a)
        VALID_MAX_LEN_A="$2"
        shift
    ;;
    --valid-max-len-b)
        VALID_MAX_LEN_B="$2"
        shift
    ;;
    --save-every-n)
        SAVE_EVERY_N_UPDATES="$2"
        shift
    ;;
    --reset-optimizer)
        RESET_OPTIMIZER_OPT="--reset-optimizer"
    ;;
    --init-ckpt)
        INIT_CKPT="$2"
        shift
    ;;
    --shared-dict)
        SHARED_DICT_OPT="--share-all-embeddings"
    ;;
    --overwrite)
        OVERWRITE=0
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
#MODEL_DIR="$MODEL_DIR.warmup-$WARMUP"
#MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
#MODEL_DIR="$MODEL_DIR.ffn-size-$FFN_SIZE"
MODEL_DIR="$MODEL_DIR.att-heads-$ATT_HEADS"
MODEL_DIR="$MODEL_DIR.depth-$DEPTH"
MODEL_DIR="$MODEL_DIR.lr-$LR"
#MODEL_DIR="$MODEL_DIR.ctrl-depth-$CTRL_DEPTH"
#MODEL_DIR="$MODEL_DIR.max-tokens-$MAX_TOKENS"
[[ -z "$RESET_OPTIMIZER_OPT" ]] || MODEL_DIR="$MODEL_DIR.reset-optim"
[[ -z "$CTRL_AVG_TOKENS_OPT" ]] || MODEL_DIR="$MODEL_DIR.mod-avg-tokens"
[[ -z "$CTRL_SAMPLING_OPT" ]] || MODEL_DIR="$MODEL_DIR.mod-hard-samples"
MODEL_DIR="$MODEL_DIR.anneal-$CTRL_ANNEAL_TYPE"
MODEL_DIR="$MODEL_DIR.anneal-rate-$CTRL_ANNEAL_RATE"
MODEL_DIR="$MODEL_DIR.ratio-$CTRL_BUDGET_RATIO"
MODEL_DIR="$MODEL_DIR.max-temp-$CTRL_MAX_TEMP"
MODEL_DIR="$MODEL_DIR.reg-$CTRL_BUDGET_WEIGHT"
MODEL_DIR="${MODEL_DIR}${EXP_SUFFIX}"


[[ -d $MODEL_DIR ]] && [[ $OVERWRITE -eq 0 ]] && rm -r $MODEL_DIR
mkdir $MODEL_DIR
echo $TASKS | sed 's/ /->/g' > $MODEL_DIR/TASKS

ckpt_opt=
[[ -n "$INIT_CKPT" ]] && ckpt_opt="--restore-file $INIT_CKPT"

epochs=$EPOCHS
valid_sets=`echo $VALID_TASKS | sed 's/ /.15.valid,/g;s/$/.15.valid/'`

for current_task in $TASKS; do
    echo Training $current_task...

    MODULE_MASK_OPT=
    if [[ -e "$EXPDIR/data/${current_task}.mask" ]]; then
        MODULE_MASK_OPT="--module-ctrl-fixed-mask $(cat $EXPDIR/data/${current_task}.mask)"
    fi
    PARAM_FREEZE_OPT=
    if [[ -e "$EXPDIR/data/${current_task}.freeze" ]]; then
        PARAM_FREEZE_OPT="--parameter-freeze-substr $(cat $EXPDIR/data/${current_task}.freeze)"
    fi

    jid=$(qsubmit \
        --queue="gpu-troja.q" \
        --logdir=$MODEL_DIR/logs \
        --jobname=tformer_masking \
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
            --eval-bleu-args '{\"beam\": $VALID_BEAM_SIZE, \"max_len_a\": $VALID_MAX_LEN_A, \"max_len_b\": $VALID_MAX_LEN_B}' \
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
            $CTRL_SAMPLING_OPT \
            $CTRL_AVG_TOKENS_OPT \
            $CTRL_OUTPUT_BIAS_OPT \
            --module-ctrl-max-temperature $CTRL_MAX_TEMP \
            --module-ctrl-min-temperature $CTRL_MIN_TEMP \
            --module-ctrl-anneal-type $CTRL_ANNEAL_TYPE \
            --module-ctrl-anneal-rate $CTRL_ANNEAL_RATE \
            --module-ctrl-cosine-reset-decay $CTRL_COSINE_DECAY \
            --module-ctrl-cosine-reset-every-n-epochs $CTRL_COSINE_RESET \
            --module-kl-div-regularizer-ratio $CTRL_KL_DIV_RATIO \
            --module-kl-div-regularizer-weight $CTRL_KL_DIV_WEIGHT \
            --module-budget-regularizer-ratio $CTRL_BUDGET_RATIO \
            --module-budget-regularizer-weight $CTRL_BUDGET_WEIGHT \
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

    echo "Finished $current_task. Evaluating..."
    for eval_task in $TASKS; do
        EVAL_MASK_OPT=
        if [[ -e "$EXPDIR/data/${eval_task}.mask" ]]; then
            EVAL_MASK_OPT="--module-mask $(cat $EXPDIR/data/${eval_task}.mask)"
        fi
        bash process_checklist.sh \
            -t $current_task \
            --expdir $MODEL_DIR \
            --tasks "$VALID_TASKS" \
            --eval-dir $EVAL_DIR \
            $EVAL_MASK_OPT \
            --translation-options '--print-module-mask --print-module-probs' | \
            tee -a $MODEL_DIR/logs/$eval_task.eval.log &
    done
    epochs=`expr $epochs + $EPOCHS`
done
