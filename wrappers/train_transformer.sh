#~/bin/bash

set -e
#set -x

JOB_PRIORITY=-90

EXPDIR=
EMB_SIZE=512
FFN_SIZE=2048
ENC_ATT_HEADS=8
DEC_ATT_HEADS=8

LR="5e-4"
MAX_TOKENS=4096
WARMUP=4000
CLIP_NORM=0.0
PATIENCE=30
KEEP_N_CHECKPOINTS=1
SAVE_EVERY_N_UPDATES=0
#SAVE_EVERY_N_UPDATES=100000
#LR=0.00024
#MAX_TOKENS=2048

RANDOM_SEED=42

HELP=0
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--expdir)
        EXPDIR="$2"
        shift
    ;;
    -r|--random-seed)
        RANDOM_SEED="$2"
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
    --enc-att-heads)
        ENC_ATT_HEADS="$2"
        shift
    ;;
    --dec-att-heads)
        DEC_ATT_HEADS="$2"
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
    --save-every-n)
        SAVE_EVERY_N_UPDATES="$2"
        shift
    ;;
    -h|--help)
        HELP=1
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

# small: 4000 warmup, 5e-4 lr
MODEL_DIR="$EXPDIR/transformer"
MODEL_DIR="$MODEL_DIR.seed-$RANDOM_SEED"
MODEL_DIR="$MODEL_DIR.warmup-$WARMUP"
MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
MODEL_DIR="$MODEL_DIR.ff-size-$FFN_SIZE"
MODEL_DIR="$MODEL_DIR.dec-att-heads-$DEC_ATT_HEADS"
MODEL_DIR="$MODEL_DIR.enc-att-heads-$ENC_ATT_HEADS"
MODEL_DIR="$MODEL_DIR.lr-$LR"
MODEL_DIR="$MODEL_DIR.max-tokens-$MAX_TOKENS"
[[ -d $MODEL_DIR ]] && rm -r $MODEL_DIR

qsubmit \
    --queue="gpu-troja.q" \
    --logdir=$MODEL_DIR/logs \
    --jobname=tformer_base \
    --mem=25g \
    --cores=4 \
    --gpumem=11g \
    --priority=$JOB_PRIORITY \
    --gpus=1 "source $HOME/python-virtualenv/fairseq-env/bin/activate && \
        fairseq-train \
        $EXPDIR/data \
        --seed $RANDOM_SEED \
        --task translation \
        --arch transformer \
        --share-decoder-input-output-embed \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm $CLIP_NORM \
        --patience $PATIENCE \
        --lr $LR \
        --lr-scheduler inverse_sqrt \
        --warmup-updates $WARMUP \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --max-tokens 2048 \
        --eval-bleu \
        --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --maximize-best-checkpoint-metric \
        --best-checkpoint-metric bleu \
        --tensorboard-logdir $MODEL_DIR \
        --save-dir $MODEL_DIR/checkpoints \
        --keep-last-epochs $KEEP_N_CHECKPOINTS \
        --encoder-attention-heads $ENC_ATT_HEADS \
        --decoder-attention-heads $DEC_ATT_HEADS \
        --encoder-embed-dim $EMB_SIZE \
        --encoder-ffn-embed-dim $FFN_SIZE \
        --save-interval-updates $SAVE_EVERY_N_UPDATES"
