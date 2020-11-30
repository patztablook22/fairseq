#!/bin/bash

. /home/aires/personal_work_troja/marian_task/marian-task/bin/activate

curr_date=`date +%s`
chckpnt=checkpoints_en-cs_$curr_date
tnsrbrd=tensorboard_logs_$curr_date

if [ -d "$chckpnt" ]; then
    rm -r $chckpnt
fi

if [ -d "$tnsrbrd" ]; then
    rm -r $tnsrbrd
fi


fairseq-train \
    data-bin/czeng \
    --arch transformer_vaswani_wmt_en_de_big --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.99, 0.998)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --tensorboard-logdir $tnsrbrd \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1500 \
    --save-dir $chckpnt \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

fairseq-generate data-bin/czeng \
    --path $chckpnt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
