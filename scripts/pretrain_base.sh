#!/bin/bash

MODEL=$1  # graph encoder: gemnet-oc, et, escn, adsgt
[ ! -d "exp_results/pretrain/base" ] && mkdir -p exp_results/pretrain/base

########### Single GPU  ###########
python -u main.py \
    --mode pretrain \
    --config-yml configs/pretrain/base/${MODEL}.yml \
    --identifier pretrain_base_${MODEL} \
    > exp_results/pretrain/base/${MODEL}.log 2>&1

###########  For multiple GPUs  ###########
# GPU_NUM=4
# python -u -m torch.distributed.launch --nproc_per_node=${GPU_NUM} main.py \
#     --mode pretrain --config-yml configs/pretrain/$DATASET/${MODEL}.yml \
#     --identifier pretrain_${DATASET}_${MODEL} \
#     --num-gpus ${GPU_NUM} --distributed \
#     > exp_results/pretrain/base/${MODEL}.log 2>&1
