#!/bin/bash
# ==============================================================================
# ABLATION arm B: HSFPN 'dw' + LWGA decoder
#   Single-variable ablation against the sibling HSFPN + LWGA run:
#     arm A (baseline): run_deimv2_hgnetv2_n_uavdt_HSFPN_lwga.sh -> hsfpn_out_mode 'dw_pw'
#     arm B (this)    : run_deimv2_hgnetv2_n_uavdt_HSFPN_dw_lwga.sh -> hsfpn_out_mode 'dw'
#   The ONLY delta is the HFP output projection; everything else matches arm A
#   (HGNetv2 B0 backbone, HFP @ stride-16, LWGA decoder @ stage-3 + PaQ aux loss).
#   Trained from scratch (no -r resume). GPU 4,5 / port 29506 (arm A uses 0,1 /
#   29500, so they can run concurrently).
#   Dependency: the DCT path requires `torch_dct` (pip install torch_dct).
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29506  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_HSFPN_dw_LWGA.yml \
    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_n_uavdt_HSFPN_dw_LWGA/last.pth \
    --use-amp --seed=0
