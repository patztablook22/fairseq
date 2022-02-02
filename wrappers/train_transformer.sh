#~/bin/bash
set -e

JOB_PRIORITY=-70
GPUMEM="11g"

EXPDIR=
EVAL_DIR=
TASKS="czeng"
VALID_TASKS="newstest"

SRC=en
TGT=cs

# General Architecture Details
EMB_SIZE=512
FFN_SIZE=$(expr 4 \* $EMB_SIZE)
ATT_HEADS=8
DEPTH=6
SHARED_DICT_OPT=

# Training Reset
INIT_CKPT=
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

# Validation - Beam Search Details
VALID_BEAM_SIZE=1
VALID_MAX_LEN_A=1.2
VALID_MAX_LEN_B=10
VALID_LENPEN=0.6

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
    -v|--valid-tasks)
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
        # unknown option
    ;;
esac
shift
done

[[ -d "$EXPDIR" ]] || exit 1
[[ -d "$EVAL_DIR" ]] || exit 1

# TODO: print help

MODEL_DIR="$EXPDIR/transformer"
MODEL_DIR="$MODEL_DIR.seed-$RANDOM_SEED"
MODEL_DIR="$MODEL_DIR.tasks-`echo $TASKS | sed 's/,/-/g'`"
MODEL_DIR="$MODEL_DIR.warmup-$WARMUP"
MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
MODEL_DIR="$MODEL_DIR.att-heads-$ATT_HEADS"
MODEL_DIR="$MODEL_DIR.depth-$DEPTH"
MODEL_DIR="$MODEL_DIR.patience-$PATIENCE"
MODEL_DIR="$MODEL_DIR.lr-$LR"
#MODEL_DIR="$MODEL_DIR.max-tokens-$MAX_TOKENS"
[[ -z "$RESET_OPTIMIZER_OPT" ]] || MODEL_DIR="$MODEL_DIR.reset-optim"

[[ -d $MODEL_DIR ]] && rm -r $MODEL_DIR
mkdir $MODEL_DIR
echo $TASKS | sed 's/ /->/g' > $MODEL_DIR/TASKS

ckpt_opt=
[[ -e "$INIT_CKPT" ]] && ckpt_opt="--restore-file $INIT_CKPT"
epochs=$EPOCHS
valid_sets=`echo $VALID_TASKS | sed 's/ /.valid,/g;s/$/.valid/'`
for current_task in $TASKS; do
    echo Training $current_task...
    jid=$(qsubmit \
        --queue="gpu-troja.q" \
        --logdir=$MODEL_DIR/logs \
        --jobname=$current_task.train \
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
                --task translation \
                --arch transformer \
                --share-decoder-input-output-embed \
                $SHARED_DICT_OPT \
                --train-subset ${current_task}.train \
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
                --criterion label_smoothed_cross_entropy \
                --label-smoothing $LABEL_SMOOTHING \
                --max-tokens $MAX_TOKENS \
                --eval-bleu \
                --eval-bleu-args '{\"beam\": $VALID_BEAM_SIZE, \"max_len_a\": $VALID_MAX_LEN_A, \"max_len_b\": $VALID_MAX_LEN_B, \"lenpen\": $VALID_LENPEN}' \
                --eval-bleu-detok moses \
                --eval-bleu-remove-bpe \
                --eval-bleu-print-samples \
                --best-checkpoint-metric bleu \
                --maximize-best-checkpoint-metric \
                --tensorboard-logdir $MODEL_DIR \
                --save-dir $MODEL_DIR/checkpoints \
                --keep-last-epochs $KEEP_N_CHECKPOINTS \
                --encoder-attention-heads $ATT_HEADS \
                --decoder-attention-heads $ATT_HEADS \
                --encoder-layers $DEPTH \
                --decoder-layers $DEPTH \
                --encoder-embed-dim $EMB_SIZE \
                --encoder-ffn-embed-dim $FFN_SIZE \
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
    bash process_checklist.sh \
        -t $current_task \
        --src $SRC \
        --tgt $TGT \
        --expdir $MODEL_DIR \
        --tasks "newstest" \
        --eval-dir $EVAL_DIR \
        --eval-prefix "test" &

    epochs=`expr $epochs + $EPOCHS`
done
