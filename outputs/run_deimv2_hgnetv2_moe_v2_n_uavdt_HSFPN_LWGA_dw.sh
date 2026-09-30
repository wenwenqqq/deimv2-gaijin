#!/bin/bash
# ==============================================================================
# ABLATION arm B (3-way combo): HFP output projection = 'dw' (dw3x3 only + GN)
#   Single-variable ablation against the sibling 3-way combo run:
#     arm A (baseline): run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA.sh
#                       -> hsfpn_out_mode 'dw_pw'  (GPU 0,1 / port 29503)
#     arm B (this)    : run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw.sh
#                       -> hsfpn_out_mode 'dw'     (GPU 6,7 / port 29505)
#   The ONLY delta is the HFP output projection (MoE v2 + HSFPN + LWGA are
#   otherwise identical). Trained from scratch (no -r resume). GPU 6,7 / port
#   29505 so it can run concurrently with arm A.
#   Dependency: the DCT path requires `torch_dct` (pip install torch_dct).
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29505  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw.yml \
    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw/last.pth \
    --use-amp --seed=0
