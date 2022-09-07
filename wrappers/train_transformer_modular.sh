#!/bin/bash
set -e

JOB_PRIORITY=-60
GPUMEM="11g"

EXPDIR=
EVAL_DIR=
EXP_SUFFIX=
TASKS="czeng"
#TASKS="id push pop shift unshift reverse"
VALID_TASKS="bpe.newstest"
#VALID_TASKS=$TASKS

SRC=en
TGT=cs

# General Architecture Details
EMB_SIZE=128
FFN_SIZE=$(expr 4 \* $EMB_SIZE)
ATT_HEADS=8
DEPTH=1
SHARED_DICT_OPT=

# Training Reset
INIT_CKPT=
RESET_OPTIMIZER_OPT=

# Training Details
RANDOM_SEED=42
INITIAL_EPOCH=0  # used with initial checkpoint (which have non-zero epoch number)
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

ADAM_BETA_1=0.9
ADAM_BETA_2=0.98

# Validation - Beam Search Details
VALID_BEAM_SIZE=1
VALID_MAX_LEN_A=1.2
VALID_MAX_LEN_B=10
VALID_LENPEN=0.6

BEST_METRIC="bleu"  # bleu, ter, accuracy
MINIMIZE_METRIC="--maximize-best-checkpoint-metric"  # We want maximization to be default

EVAL_SCRIPT=process_checklist.sh

# Modular Details
CTRL_TYPE="attention"  # which modules are controlled (attention, ffn, TODO)
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

# Parameter freeze (for consecutive tasks)
FREEZE_PARAMS=

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
    --src)
        SRC="$2"
        shift
    ;;
    --tgt)
        TGT="$2"
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
    --initial-epoch)
        INITIAL_EPOCH="$2"
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
    --valid-lenpen)
        VALID_LENPEN="$2"
        shift
    ;;
    --adam-beta-1)
        ADAM_BETA_1="$2"
        shift
    ;;
    --adam-beta-2)
        ADAM_BETA_2="$2"
        shift
    ;;
    --freeze-params)
        FREEZE_PARAMS="$2"
        shift
    ;;
    --eval-script)
        EVAL_SCRIPT="$2"
        shift
    ;;
    --best-metric)
        BEST_METRIC="$2"
        shift
    ;;
    --minimize-metric)
        MINIMIZE_METRIC=""
    ;;
    --save-every-n)
        SAVE_EVERY_N_UPDATES="$2"
        shift
    ;;
    --init-ckpt)
        INIT_CKPT="$2"
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
#MODEL_DIR="$MODEL_DIR.warmup-$WARMUP"
#MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
MODEL_DIR="$MODEL_DIR.att-heads-$ATT_HEADS"
MODEL_DIR="$MODEL_DIR.depth-$DEPTH"
MODEL_DIR="$MODEL_DIR.smooth-$LABEL_SMOOTHING"
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


[[ -d $MODEL_DIR ]] && rm -r $MODEL_DIR
mkdir $MODEL_DIR && mkdir $MODEL_DIR/checkpoints
echo $TASKS | sed 's/ /->/g' > $MODEL_DIR/TASKS
echo $FREEZE_PARAMS > $MODEL_DIR/FREEZE_PARAMS

ckpt_opt=
[[ -e "$INIT_CKPT" ]] && ckpt_opt="--restore-file $INIT_CKPT" && cp $INIT_CKPT $MODEL_DIR/checkpoints/checkpoint_last.pt
epochs=`expr $INITIAL_EPOCH + $EPOCHS`
valid_sets=`echo $VALID_TASKS | sed 's/ /.valid,/g;s/$/.valid/'`
for current_task in $TASKS; do
    echo Training $current_task...

    # We freeze the parameters only after finishing the training of the first task
    PARAM_FREEZE_OPT=
    if [[ -e "$MODEL_DIR/checkpoints/checkpoint_last.pt" ]] || [[ -e "$INIT_CKPT" ]]; then
        [[ -n "$FREEZE_PARAMS" ]] && PARAM_FREEZE_OPT="--parameter-freeze-substr '$FREEZE_PARAMS'"
    fi

    jid=$(qsubmit \
        --queue="gpu-troja.q" \
        --logdir=$MODEL_DIR/logs \
        --jobname=$current_task.train.mod \
        --mem=25g \
        --cores=4 \
        --gpumem=$GPUMEM \
        --priority=$JOB_PRIORITY \
        --gpus=1 "source $HOME/python-virtualenv/fairseq-env/bin/activate && \
            export CUDA_LAUNCH_BLOCKING=1 && \
            python train.py \
                $EXPDIR/data \
                -s $SRC \
                -t $TGT \
                --seed $RANDOM_SEED \
                --task translation_modular \
                --arch transformer_modular \
                --share-decoder-input-output-embed \
                $SHARED_DICT_OPT \
                --train-subset ${current_task}.train \
                --valid-subset $valid_sets \
                $ckpt_opt \
                $RESET_OPTIMIZER_OPT \
                --optimizer adam \
                --adam-betas '($ADAM_BETA_1, $ADAM_BETA_2)' \
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
                --eval-bleu-args '{\"beam\": $VALID_BEAM_SIZE, \"max_len_a\": $VALID_MAX_LEN_A, \"max_len_b\": $VALID_MAX_LEN_B, \"lenpen\": $VALID_LENPEN}' \
                --eval-bleu-detok moses \
                --eval-bleu-remove-bpe \
                --eval-bleu-print-samples \
                --best-checkpoint-metric $BEST_METRIC \
                $MINIMIZE_METRIC \
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
                --module-ctrl-cosine-reset-every-n-steps $CTRL_COSINE_RESET \
                --module-kl-div-regularizer-ratio $CTRL_KL_DIV_RATIO \
                --module-kl-div-regularizer-weight $CTRL_KL_DIV_WEIGHT \
                --module-budget-regularizer-ratio $CTRL_BUDGET_RATIO \
                --module-budget-regularizer-weight $CTRL_BUDGET_WEIGHT \
                $PARAM_FREEZE_OPT \
                --save-interval-updates $SAVE_EVERY_N_UPDATES")

    jid=`echo $jid | cut -d" " -f3`
    echo Waiting for $jid...
    while true; do
        sleep 15
        qstat | grep $jid > /dev/null || break
    done

    # Save the last ckpt for current task and set it as initial
    # ckpt for the next iteration
    cp $MODEL_DIR/checkpoints/checkpoint_last.pt \
        $MODEL_DIR/checkpoints/checkpoint_$current_task.pt
    ckpt_opt="--restore-file $MODEL_DIR/checkpoints/checkpoint_$current_task.pt"

    echo "Finished $current_task. Evaluating..."
    for t in valid test; do
        bash $EVAL_SCRIPT \
            -t $current_task \
            --src $SRC \
            --tgt $TGT \
            --expdir $MODEL_DIR \
            --tasks "`echo $VALID_TASKS | sed 's/\.[^. ]* / /;s/\.[^. ]*$//'`" \
            --eval-dir $EVAL_DIR \
            --eval-prefix "$t" \
            --translation-options '--print-module-mask --print-module-probs' \
        | tee -a $MODEL_DIR/logs/$current_task.eval.log &
    done

    epochs=`expr $epochs + $EPOCHS`
done
