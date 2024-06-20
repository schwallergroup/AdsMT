#!/bin/bash

DATASET=$1
MODEL=$2  # graph encoder: gemnet-oc, et, escn, adsgt
[ ! -d "exp_results/pretrain/$DATASET" ] && mkdir -p exp_results/pretrain/$DATASET

########### Single GPU  ###########
python -u main.py \
    --mode pretrain \
    --config-yml configs/pretrain/$DATASET/${MODEL}.yml \
    --identifier pretrain_${DATASET}_${MODEL} \
    > exp_results/pretrain/$DATASET/${MODEL}.log 2>&1

###########  For multiple GPUs  ###########
# GPU_NUM=4
# python -u -m torch.distributed.launch --nproc_per_node=${GPU_NUM} main.py \
#     --mode pretrain --config-yml configs/pretrain/$DATASET/${MODEL}.yml \
#     --identifier pretrain_${DATASET}_${MODEL} \
#     --num-gpus ${GPU_NUM} --distributed \
#     > exp_results/pretrain/$DATASET/${MODEL}.log 2>&1
