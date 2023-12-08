#!/bin/bash


EXP_DIR=$SCRATCH/experiments/str-edit.compo.30000
MODEL=transformer.seed-42.emb-size-128.att-heads-1.depth-1

ls $EXP_DIR
if [ ! -d $EXP_DIR/$MODEL ]; then
    echo "No such directory: $EXP_DIR/$MODEL"
    exit 1
fi

TARGET="$SCRATCH/export_$(date +%s)"
SOURCE=$EXP_DIR/$MODEL

mkdir -p $TARGET

cp -r $SOURCE/train $TARGET
cp -r $SOURCE/train_inner $TARGET
cp -r $SOURCE/*.valid $TARGET

echo "TARGET:"
echo $TARGET
