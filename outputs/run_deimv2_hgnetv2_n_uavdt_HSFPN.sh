#!/bin/bash
# ==============================================================================
# DEIMv2 + HGNetv2 B0 on UAVDT with HSFPN encoder
#   HFP replaces intra-scale TE at stride-16 with Lightweight-HFP (dw_pw+share_dct).
#   Resume from last checkpoint.
#
#   ABLATION PAIR (arm A / baseline): this run uses hsfpn_out_mode='dw_pw'.
#     arm B (sibling): run_deimv2_hgnetv2_n_uavdt_HSFPN_dw.sh  -> hsfpn_out_mode='dw'
#       (drops the pw1x1 channel mix of the HFP output projection; everything
#        else identical). The ONLY delta is hsfpn_out_mode. arm B uses GPU 6,7 /
#        port 29504 so both can run concurrently with this one (GPU 2,3 / 29502).
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_HSFPN.yml \
    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_n_uavdt_HSFPN_dw_pw_real_all/last.pth \
    --use-amp --seed=0
