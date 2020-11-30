#!/bin/bash

set -e
#set -x

JOB_PRIORITY=-110

EXPDIR=
EMB_SIZE=512
FFN_SIZE=2048
ENC_ATT_HEADS=8
ENC_ATT_HEADS_ACTIVE=4
DEC_ATT_HEADS=8
DEC_ATT_HEADS_ACTIVE=4
ENC_MODULAR_LAYERS=""
DEC_MODULAR_LAYERS=""

LR="5e-4"
MAX_TOKENS=4096
WARMUP="4000"
CLIP_NORM=0.0
PATIENCE=30
KEEP_N_CHECKPOINTS=1
SAVE_EVERY_N_UPDATES=0
#SAVE_EVERY_N_UPDATES=100000
#LR=0.00024
#MAX_TOKENS=2048
STOP_CRIT="ctrl_loss"


DROPOUT=0.3
RANDOM_SEED=42
N_SAMPLES=10
M_STEPS=10
CTRL_ALPHA=1.0
CTRL_DEPTH=0
CTRL_DIM=2048
CTRL_DROP=0.0
CTRL_TYPE="joint"
SHARE_CTRL_OPT=""
HP_KEYS="module_ctrl_hidden_depth,encoder_attention_heads_active,decoder_attention_heads_active,e_step_size,m_steps,clip_norm,dropout,module_ctrl_word_dropout,share_encoder_ctrl"

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
    --enc-att-heads-active)
        ENC_ATT_HEADS_ACTIVE="$2"
        shift
    ;;
    --dec-att-heads)
        DEC_ATT_HEADS="$2"
        shift
    ;;
    --dec-att-heads-active)
        DEC_ATT_HEADS_ACTIVE="$2"
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
    --enc-modular-layers)
        ENC_MODULAR_LAYERS="$2"
        shift
    ;;
    --dec-modular-layers)
        DEC_MODULAR_LAYERS="$2"
        shift
    ;;
    --ctrl-alpha)
        CTRL_ALPHA="$2"
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
    --ctrl|--controller-type)
        CTRL_TYPE="$2"
        shift
    ;;
    --stop-crit)
        STOP_CRIT="$2"
        shift
    ;;
    --share-ctrl|--share-controller)
        SHARE_CTRL_OPT="--share-encoder-ctrl"
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


enc_mod_layers=`echo $ENC_MODULAR_LAYERS | sed 's/,/-/g'`
dec_mod_layers=`echo $DEC_MODULAR_LAYERS | sed 's/,/-/g'`


# small: 4000 warmup, 5e-4 lr
MODEL_DIR="$EXPDIR/transformer_modular"
MODEL_DIR="$MODEL_DIR.seed-$RANDOM_SEED"
#MODEL_DIR="$MODEL_DIR.warmup-$WARMUP"
MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
#MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
#MODEL_DIR="$MODEL_DIR.dec-att-heads-$DEC_ATT_HEADS"
#MODEL_DIR="$MODEL_DIR.enc-att-heads-$ENC_ATT_HEADS"
MODEL_DIR="$MODEL_DIR.enc-att-active-$ENC_ATT_HEADS_ACTIVE"
MODEL_DIR="$MODEL_DIR.dec-att-active-$DEC_ATT_HEADS_ACTIVE"
MODEL_DIR="$MODEL_DIR.ctrl-type-$CTRL_TYPE"
[[ -z "$SHARE_CTRL_OPT" ]] || MODEL_DIR="$MODEL_DIR.shared-ctrl"
MODEL_DIR="$MODEL_DIR.n-samples-$N_SAMPLES"
MODEL_DIR="$MODEL_DIR.m-steps-$M_STEPS"
#MODEL_DIR="$MODEL_DIR.ctrl-alpha-$CTRL_ALPHA"
#MODEL_DIR="$MODEL_DIR.lr-$LR"
#MODEL_DIR="$MODEL_DIR.max-tokens-$MAX_TOKENS"
MODEL_DIR="$MODEL_DIR.ctrl-depth-$CTRL_DEPTH"
#MODEL_DIR="$MODEL_DIR.ctrl-dim-$CTRL_DIM"
MODEL_DIR="$MODEL_DIR.ctrl-drop-$CTRL_DROP"
if [ "$enc_mod_layers" == "$dec_mod_layers" ]; then
    MODEL_DIR="$MODEL_DIR.mod-layers-$enc_mod_layers"
else
    MODEL_DIR="$MODEL_DIR.enc-mod-layers-$enc_mod_layers"
    MODEL_DIR="$MODEL_DIR.dec-mod-layers-$dec_mod_layers"
fi
[[ -d $MODEL_DIR ]] && rm -r $MODEL_DIR


#qsubmit \
#    --queue="gpu-troja.q" \
qsubmit \
    --logdir=$MODEL_DIR/logs \
    --jobname=tformer_mod \
    --mem=25g \
    --cores=4 \
    --gpumem=11g \
    --priority=$JOB_PRIORITY \
    --gpus=1 "source $HOME/python-virtualenv/fairseq-env/bin/activate && \
        export CUDA_LAUNCH_BLOCKING=1 && \
        fairseq-train \
        $EXPDIR/data \
        --seed $RANDOM_SEED \
        --task translation_modular \
        --arch transformer_modular \
        --share-decoder-input-output-embed \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm $CLIP_NORM \
        --patience $PATIENCE \
        --lr $LR \
        --lr-scheduler inverse_sqrt \
        --warmup-updates $WARMUP \
        --dropout $DROPOUT \
        --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy_modular \
        --label-smoothing 0.1 \
        --max-tokens $MAX_TOKENS \
        --eval-bleu \
        --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric $STOP_CRIT \
        --tensorboard-logdir $MODEL_DIR \
        --tensorboard-hparams-keys \"$HP_KEYS\" \
        --save-dir $MODEL_DIR/checkpoints \
        --keep-last-epochs $KEEP_N_CHECKPOINTS \
        --encoder-attention-heads $ENC_ATT_HEADS \
        --encoder-attention-heads-active $ENC_ATT_HEADS_ACTIVE \
        --decoder-attention-heads $DEC_ATT_HEADS \
        --decoder-attention-heads-active $DEC_ATT_HEADS_ACTIVE \
        --encoder-embed-dim $EMB_SIZE \
        --encoder-ffn-embed-dim $FFN_SIZE \
        --encoder-modular-layer-indices \"($ENC_MODULAR_LAYERS)\" \
        --decoder-modular-layer-indices \"($DEC_MODULAR_LAYERS)\" \
        --e-step-size $N_SAMPLES \
        --m-steps $M_STEPS \
        --ctrl-alpha $CTRL_ALPHA \
        --module-ctrl-hidden-depth $CTRL_DEPTH \
        --module-ctrl-hidden-dim $CTRL_DIM \
        --module-ctrl-word-dropout $CTRL_DROP \
        --module-ctrl-type \"$CTRL_TYPE\" \
        $SHARE_CTRL_OPT \
        --save-interval-updates $SAVE_EVERY_N_UPDATES"
