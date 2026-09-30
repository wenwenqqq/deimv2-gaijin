#!/bin/bash
# ==============================================================================
# DEIMv2 + HGNetv2 B0 on UAVDT with Combo encoder (HFP + PFG)
#   HFP @ stride-16 (high-res DCT high-pass for tiny objects)
#   PFG @ stride-32 (deep large-RF frequency-gated token mixing)
#   Train from scratch.
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_Combo.yml \
    --use-amp --seed=0
