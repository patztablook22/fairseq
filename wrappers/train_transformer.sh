#~/bin/bash
set -e

JOB_PRIORITY=10
SLURM_CONSTRAINTS=
EXCLUDE_NODES="dll-10gpu1,dll-10gpu2,dll-10gpu3"
SLURM_SUBMIT="/home/varis/scripts/slurm-submit.sh"

VIRTUALENV="/home/varis/python-virtualenv/fairseq-env/bin/activate"
CORES=4
MEM="25g"
GPUMEM="11g"
GPUS=1

EXPDIR=
EVAL_DIR=
TASKS="czeng"
#TASKS="id.15 push.15 pop.15 shift.15 unshift.15 reverse.15"
VALID_TASKS="bpe.newstest"
#VALID_TASKS=$TASKS

SRC=en
TGT=cs

# General Architecture Details
EMB_SIZE=512
FFN_SIZE=$(expr 4 \* $EMB_SIZE)
ATT_HEADS=8
DEPTH=6
SHARED_DICT_OPT=
TIE_DECODER_OPT=

# Training Reset
INIT_CKPT=
RESET_OPTIMIZER_OPT=

# Training Details
RANDOM_SEED=42
N_UPDATES=0
INITIAL_UPDATE=0  # used with initial checkpoint (which have non-zero epoch number)
UPDATE_OPT="--max-update"
LABEL_SMOOTHING=0.1
MAX_TOKENS=4096
DROPOUT=0.2
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

EVAL_SCRIPT=process_checklist.sh  # Validation wrapper (called at the end of each task training)

# EWC
EWC_LAMBDA="0."
EWC_TERM_TYPE="original"  # original, norm_1, norm_2
EWC_EST_SUBSET="valid"
EWC_NORM="tokens"  # tokens, sentences, batches

# Parameter freeze (for consecutive tasks)
FREEZE_PARAMS=

OPTS="$*"  # Save script parameters for rerun...

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
    --max-updates)
        N_UPDATES="$2"
        shift
    ;;
    --epochs-not-updates)
        UPDATE_OPT="--max-epoch"
    ;;
    --initial-update)
        INITIAL_UPDATE="$2"
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
    --dropout)
        DROPOUT="$2"
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
    --ewc-lambda)
        EWC_LAMBDA="$2"
        shift
    ;;
    --ewc-term-type)
        EWC_TERM_TYPE="$2"
        shift
    ;;
    --ewc-est-subset)
        EWC_EST_SUBSET="$2"
        shift
    ;;
    --ewc-est-norm)
        EWC_NORM="$2"
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
    --tie-decoder-output)
        TIE_DECODER_OPT="--share-decoder-input-output-embed"
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
#MODEL_DIR="$MODEL_DIR.tasks-`echo $TASKS | sed 's/ /-/g'`"
MODEL_DIR="$MODEL_DIR.warmup-$WARMUP"
MODEL_DIR="$MODEL_DIR.clip-norm-$CLIP_NORM"
MODEL_DIR="$MODEL_DIR.emb-size-$EMB_SIZE"
MODEL_DIR="$MODEL_DIR.att-heads-$ATT_HEADS"
MODEL_DIR="$MODEL_DIR.depth-$DEPTH"
MODEL_DIR="$MODEL_DIR.smooth-$LABEL_SMOOTHING"
MODEL_DIR="$MODEL_DIR.patience-$PATIENCE"
MODEL_DIR="$MODEL_DIR.lr-$LR"
[[ "$EWC_LAMBDA" != "0." ]] && MODEL_DIR="$MODEL_DIR.ewc-$EWC_TERM_TYPE-$EWC_LAMBDA.ewc-est-$EWC_EST_SUBSET-$EWC_NORM"
#MODEL_DIR="$MODEL_DIR.max-tokens-$MAX_TOKENS"
[[ -z "$RESET_OPTIMIZER_OPT" ]] || MODEL_DIR="$MODEL_DIR.reset-optim"

[[ -d $MODEL_DIR ]] && rm -r $MODEL_DIR
mkdir $MODEL_DIR && mkdir $MODEL_DIR/checkpoints
echo $OPTS > $MODEL_DIR/CMD_ARGS
echo $TASKS | sed 's/ /->/g' > $MODEL_DIR/TASKS
echo $FREEZE_PARAMS > $MODEL_DIR/FREEZE_PARAMS

ckpt_opt=
[[ -e "$INIT_CKPT" ]] && cp $INIT_CKPT $MODEL_DIR/checkpoints/checkpoint_last.pt && ckpt_opt="--restore-file $MODEL_DIR/checkpoints/checkpoint_last.pt"

update_opt=""
n_updates=0
if [[ $PATIENCE -eq 0 ]]; then
    n_updates=`expr $INITIAL_UPDATE + $N_UPDATES`
    update_opt="$UPDATE_OPT $n_updates"
fi

valid_sets=`echo $VALID_TASKS | sed 's/ /.valid,/g;s/$/.valid/'`
for current_task in $TASKS; do
    echo Training $current_task...

    PARAM_FREEZE_OPT=
    if [[ -e "$MODEL_DIR/checkpoints/checkpoint_last.pt" ]] || [[ -e "$INIT_CKPT" ]]; then
        [[ -n $FREEZE_PARAMS ]] && PARAM_FREEZE_OPT="--parameter-freeze-substr '$FREEZE_PARAMS'"
    fi

    jid=$($SLURM_SUBMIT \
        --jobname "$current_task.train" \
        --constraints "$SLURM_CONSTRAINTS" \
        --exclude "$EXCLUDE_NODES" \
        --logdir "$MODEL_DIR/logs" \
        --gpus $GPUS \
        --mem $GPUMEM \
        --cores $CORES \
        --priority $JOB_PRIORITY "source $VIRTUALENV && \
            export CUDA_LAUNCH_BLOCKING=1 && \
            python train.py \
                $EXPDIR/data \
                -s $SRC \
                -t $TGT \
                --seed $RANDOM_SEED \
                --task translation \
                --arch transformer \
                $TIE_DECODER_OPT \
                $SHARED_DICT_OPT \
                --train-subset ${current_task}.train \
                --valid-subset $valid_sets \
                $ckpt_opt \
                $RESET_OPTIMIZER_OPT \
                --num-workers 0 \
                --optimizer adam \
                --adam-betas '($ADAM_BETA_1, $ADAM_BETA_2)' \
                --clip-norm $CLIP_NORM \
                --patience $PATIENCE \
                $update_opt \
                --lr $LR \
                --lr-scheduler inverse_sqrt \
                --warmup-updates $WARMUP \
                --dropout $DROPOUT \
                --weight-decay 0.0001 \
                --criterion label_smoothed_cross_entropy \
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
                --ewc-lambda $EWC_LAMBDA \
                --ewc-term-type $EWC_TERM_TYPE\
                $PARAM_FREEZE_OPT \
                --save-interval-updates $SAVE_EVERY_N_UPDATES")

    jid=`echo $jid | cut -d" " -f4`
    echo Waiting for $jid...
    while true; do
        sleep 15
        #qstat | grep $jid > /dev/null || break
        squeue | grep " $jid " > /dev/null || break
    done

    if [[ $PATIENCE -gt 0 ]]; then
        cp $MODEL_DIR/checkpoints/checkpoint_best.pt \
            $MODEL_DIR/checkpoints/checkpoint_last.pt
    fi

    # Consolidate (EWC)
    if [[ "$EWC_LAMBDA" != "0." ]]; then
        # TODO: Use valid data or a sample of train?
        echo "$EWC_LAMBDA != 0. => consolidating weights..."
        jid=$($SLURM_SUBMIT \
            --jobname "$current_task.consolidate" \
            --constraints "$SLURM_CONSTRAINTS" \
            --exclude "$EXCLUDE_NODES" \
            --logdir "$MODEL_DIR/logs" \
            --gpus $GPUS \
            --mem $GPUMEM \
            --cores $CORES \
            --priority $JOB_PRIORITY "source $HOME/python-virtualenv/fairseq-env/bin/activate && \
                export CUDA_LAUNCH_BLOCKING=1 && \
                python ewc_consolidate.py \
                    $EXPDIR/data \
                    -s $SRC \
                    -t $TGT \
                    --input <(paste $EVAL_DIR/$current_task.$EWC_EST_SUBSET.$SRC $EVAL_DIR/$current_task.$EWC_EST_SUBSET.$TGT) \
                    --path $MODEL_DIR/checkpoints/checkpoint_last.pt \
                    --ewc-normalize $EWC_NORM \
                    --buffer-size 100")
        jid=`echo $jid | cut -d" " -f4`
        echo Waiting for $jid...
        while true; do
            sleep 15
            #qstat | grep $jid > /dev/null || break
            squeue | grep " $jid " > /dev/null || break
        done

    fi

    # Save the last ckpt for current task and set it as initial
    # ckpt for the next iteration
    cp $MODEL_DIR/checkpoints/checkpoint_last.pt \
        $MODEL_DIR/checkpoints/checkpoint_$current_task.pt
    ckpt_opt="--restore-file $MODEL_DIR/checkpoints/checkpoint_$current_task.pt"

    # Validate
    echo "Validating using $EVAL_SCRIPT..."
    for t in test; do
        bash $EVAL_SCRIPT \
            -t best \
            --src $SRC \
            --tgt $TGT \
            --expdir $MODEL_DIR \
            --beam-size $VALID_BEAM_SIZE \
            --lenpen $VALID_LENPEN \
            --tasks "`echo $VALID_TASKS | sed 's/bpe.//g'`" \
            --eval-dir $EVAL_DIR \
            --eval-prefix "$t" 2>&1 \
        | tee -a $MODEL_DIR/logs/$current_task.eval.log &
    done

    if [[ $PATIENCE -eq 0 ]]; then
        n_updates=`expr $n_updates + $N_UPDATES`
        update_opt="$UPDATE_OPT $n_updates"
    fi
done
