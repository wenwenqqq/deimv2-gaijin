#!/bin/bash

source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29500  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_LWGA.yml \
    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_n_uavdt_LWGA/last.pth \
    --use-amp --seed=0
