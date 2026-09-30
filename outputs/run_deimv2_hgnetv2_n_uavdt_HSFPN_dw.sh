#!/bin/bash
# ==============================================================================
# ABLATION arm B: HFP output projection = 'dw' (depthwise 3x3 only + GN)
#   Single-variable ablation against the sibling HSFPN run:
#     arm A (baseline): run_deimv2_hgnetv2_n_uavdt_HSFPN.sh  -> hsfpn_out_mode 'dw_pw'
#     arm B (this)    : run_deimv2_hgnetv2_n_uavdt_HSFPN_dw.sh -> hsfpn_out_mode 'dw'
#   The ONLY delta is the HFP output projection; everything else matches arm A.
#   Trained from scratch (no -r resume). GPU 6,7 / port 29504 (arm A uses 2,3 /
#   29502, so they can run concurrently).
#   Dependency: the DCT path requires `torch_dct` (pip install torch_dct).
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29504  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_HSFPN_dw.yml \
    --use-amp --seed=0
