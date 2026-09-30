#!/bin/bash
# ==============================================================================
# DEIMv2 + HGNetv2 B0  MoE v2 backbone + HSFPN encoder + LWGA decoder on UAVDT
#   Combined 3-way experiment (from scratch):
#     * backbone : HGNetv2_MoE_v2  (block-level ES-MoE v2 @ stage4)
#     * encoder  : HybridEncoder_HSFPN (HFP @ stride-16, dw_pw, shared DCT)
#     * decoder  : DEIMTransformer_LWGA (stage-3 LWGA, kernel 11) + PaQ aux loss
#
#   Dependency: the DCT path requires `torch_dct` (pip install torch_dct).
#
#   Trained from scratch (no -r resume). 2 GPUs, port 29503 (avoids the ports
#   used by the running moev2 / HSFPN / lwga experiments).
#
#   ABLATION PAIR (arm A / baseline): this run uses hsfpn_out_mode='dw_pw'.
#     arm B (sibling): run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw.sh
#       -> hsfpn_out_mode='dw' (drops the pw1x1 channel mix of the HFP output
#          projection; MoE v2 + HSFPN + LWGA otherwise identical). The ONLY delta
#          is hsfpn_out_mode. arm B uses GPU 6,7 / port 29505 so both run
#          concurrently with this one (GPU 0,1 / 29503).
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29503  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA.yml \
    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA/last.pth \
    --use-amp --seed=0
