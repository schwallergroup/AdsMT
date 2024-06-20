#!/bin/bash

# three GMAE datasets: Alloy-GMAE, FG-GMAE, OCD-GMAE
DATASET=$1
# graph encoder: schnet, cgcnn, dpp, gemnet-oc, et, escn, adsgt
MODEL=$2 

[ ! -d "exp_results/$DATASET" ] && mkdir -p exp_results/$DATASET

python -u main.py \
    --mode train \
    --runs 10 \
    --config-yml configs/${DATASET}/${MODEL}.yml \
    --identifier train_${DATASET}_${MODEL} \
    > exp_results/${DATASET}/${MODEL}.log 2>&1
