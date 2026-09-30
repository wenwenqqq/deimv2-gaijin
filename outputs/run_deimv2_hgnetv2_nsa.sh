#!/bin/bash

source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501  train.py \
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_NSA.yml \
    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_n_uavdt_NSA/last.pth \
    --use-amp --seed=0
