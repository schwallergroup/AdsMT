#!/bin/bash

CONFIG_PATH=$1  # the path of config file of the trained AdsMT model
CHECKPOINT_PATH=$2  # the checkponit path of a trained AdsMT model

[ ! -d "exp_results/attn4sites" ] && mkdir -p exp_results/attn4sites

python -u main.py \
          --mode attn4sites \
          --seed 0 \
          --config-yml $CONFIG_PATH \
          --checkpoint $CHECKPOINT_PATH \
          --identifier attn4site \
          > exp_results/attn4sites/attn4site.log 2>&1
