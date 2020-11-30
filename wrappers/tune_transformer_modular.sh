#!/bin/bash

set -e

EXPDIR=
CKPT=
EMB_SIZE=512
FFN_SIZE=2048
ENC_ATT_HEADS=8
ENC_ATT_HEADS_ACTIVE=4
DEC_ATT_HEADS=8

RESET_OPTIMIZER_OPT=
RESET_DATALOADER_OPT=

LR="5e-4"
MAX_TOKENS=4096
WARMUP="4000"
CLIP_NORM=0.0
PATIENCE=30
KEEP_N_CHECKPOINTS=10
SAVE_EVERY_N_UPDATES=0
#SAVE_EVERY_N_UPDATES=100000
#LR=0.00024
#MAX_TOKENS=2048

N_SAMPLES=10
M_STEPS=10
MODULAR_LAYERS="0,1,2,3,4,5"
CTRL_TYPE="joint-shared"

HELP=0
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--expdir)
        EXPDIR="$2"
        shift
    ;;
    -c|--checkpoint)
        CKPT="$2"
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
    --enc-att-heads-active)
        ENC_ATT_HEADS_ACTIVE="$2"
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
    --n-samples)
        N_SAMPLES="$2"
        shift
    ;;
    --m-steps)
        M_STEPS="$2"
        shift
    ;;
    --modular-layers)
        MODULAR_LAYERS="$2"
        shift
    ;;
    --ctrl|--controller-type)
        CTRL_TYPE="$2"
        shift
    ;;
    --save-every-n)
        SAVE_EVERY_N_UPDATES="$2"
        shift
    ;;
    --reset-optimizer)
        RESET_OPTIMIZER_OPT="--reset-optimizer"
    ;;
    --reset-dataloader)
        RESET_DATALOADER_OPT="--reset-dataloader"
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


mod_layers=`echo $MODULAR_LAYERS | sed 's/,/-/g'`

# small: 4000 warmup, 5e-4 lr
MODEL_DIR="$EXPDIR/transformer_modular.warmup-$WARMUP"
MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
MODEL_DIR="$MODEL_DIR.dec-att-heads-$DEC_ATT_HEADS"
MODEL_DIR="$MODEL_DIR.enc-att-heads-$ENC_ATT_HEADS"
MODEL_DIR="$MODEL_DIR.enc-att-active-$ENC_ATT_HEADS_ACTIVE"
MODEL_DIR="$MODEL_DIR.ctrl-type-$CTRL_TYPE"
MODEL_DIR="$MODEL_DIR.n-samples-$N_SAMPLES"
MODEL_DIR="$MODEL_DIR.m-steps-$M_STEPS"
MODEL_DIR="$MODEL_DIR.lr-$LR"
MODEL_DIR="$MODEL_DIR.max-tokens-$MAX_TOKENS"
MODEL_DIR="$MODEL_DIR.mod-layers-$mod_layers"
[[ -d $MODEL_DIR ]] && rm -r $MODEL_DIR

qsubmit \
    --queue="gpu-troja.q" \
    --logdir=$MODEL_DIR/logs \
    --jobname=fairseq_mt \
    --mem=25g \
    --cores=4 \
    --gpumem=11g \
    --gpus=1 "source $HOME/python-virtualenv/fairseq-env/bin/activate && \
        fairseq-train \
        $EXPDIR/data \
        --task translation_modular \
        --arch transformer_modular \
        $RESET_OPTIMIZER_OPT \
        $RESET_DATALOADER_OPT \
        --restore-file $CKPT \
        --criterion label_smoothed_cross_entropy_modular \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm $CLIP_NORM \
        --patience $PATIENCE \
        --lr $LR \
        --lr-scheduler inverse_sqrt \
        --warmup-updates $WARMUP \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --label-smoothing 0.1 \
        --max-tokens $MAX_TOKENS \
        --eval-bleu \
        --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --maximize-best-checkpoint-metric \
        --best-checkpoint-metric bleu \
        --tensorboard-logdir $MODEL_DIR \
        --save-dir $MODEL_DIR/checkpoints \
        --share-decoder-input-output-embed \
        --keep-last-epochs $KEEP_N_CHECKPOINTS \
        --encoder-embed-dim $EMB_SIZE \
        --encoder-ffn-embed-dim $FFN_SIZE \
        --encoder-attention-heads $ENC_ATT_HEADS \
        --encoder-attention-heads-active $ENC_ATT_HEADS_ACTIVE \
        --decoder-attention-heads $DEC_ATT_HEADS \
        --encoder-modular-layer-indices \"($MODULAR_LAYERS)\" \
        --e-step-size $N_SAMPLES \
        --m-steps $M_STEPS \
        --module-ctrl-type \"$CTRL_TYPE\" \
        --save-interval-updates $SAVE_EVERY_N_UPDATES"
