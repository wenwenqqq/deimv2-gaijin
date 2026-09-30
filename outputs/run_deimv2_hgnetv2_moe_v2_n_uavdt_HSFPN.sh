#!/bin/bash
# ==============================================================================
# DEIMv2 + HGNetv2 B0 MoE v2 backbone + HSFPN encoder on UAVDT
#   MoE v2 @ stage4 backbone, HFP @ stride-16 encoder.
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN.yml \
    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN/last.pth \
    --use-amp --seed=0
