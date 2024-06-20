#!/bin/bash

DATASET=$1  # three GMAE datasets: Alloy-GMAE, FG-GMAE, OCD-GMAE
MODEL=$2  # graph encoder: gemnet-oc, et, escn, adsgt

[ ! -d "exp_results/$DATASET/finetune" ] && mkdir -p exp_results/$DATASET/finetune

python -u main.py \
    --mode train \
    --runs 10 \
    --config-yml configs/${DATASET}/finetune/${MODEL}.yml \
    --identifier finetine_${DATASET}_${MODEL} \
    > exp_results/${DATASET}/finetune/${MODEL}.log 2>&1
