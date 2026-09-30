#!/bin/bash

source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_DRFD.yml \
    --use-amp --seed=0
