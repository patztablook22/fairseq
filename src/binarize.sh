#!/bin/bash

cd "$(dirname $0)/.."

wrappers/preprocess_multiple_datasets.sh \
    --expdir $SCRATCH/experiments/str-edit.compo.30000 \
    --src x --tgt y \
    --datadir custom_examples/translation/str-edit.compo.30000 \
    --train-sets "$(for t in id flip reverse flip-reverse all; do echo $t.30; done | tr '\n' ' ')" \
    --valid-sets "$(for t in id flip reverse flip-reverse all; do for n in 20 30 40 50 60; do echo $t.$n; done; done | tr '\n' ' ')" \
    --test-sets "$(for t in id flip reverse flip-reverse all; do for n in 20 30 40 50 60; do echo $t.$n; done; done | tr '\n' ' ')" \
    --joined-dict


