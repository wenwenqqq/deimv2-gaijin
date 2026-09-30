#!/bin/bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    -c /mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_LWGA.yml \
    -r /mnt/e/Experiment/huya-deimv2/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_LWGA/last.pth \
    --use-amp --seed=0

python train.py -c configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_LWGA.yml  --use-amp --seed=0