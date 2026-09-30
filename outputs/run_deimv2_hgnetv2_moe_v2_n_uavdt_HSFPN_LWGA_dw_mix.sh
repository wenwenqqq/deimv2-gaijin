#!/bin/bash
# ==============================================================================
# NEW experiment: MoE v2 + HSFPN + LWGA_dw with a MIXED decoder
#   (decoder only uses LWGA on layers [0, 1]; the final/eval layer 2 falls back
#    to plain nn.MultiheadAttention).
#   config: configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_mix.yml
#   output: outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_mix
#
#   Sibling (baseline) all-LWGA run: ..._HSFPN_LWGA_dw  (top AP ~20.84)
#   The ONLY delta is layer-2 self-attention = MHA.
#
#   FIRST launch: -t (tuning) reuses the sibling dw run's weights
#     (backbone/encoder/cross-attn/heads). Only decoder.layers.2.self_attn is
#     freshly initialized because its state keys differ (LWGA -> MHA).
#   LATER resumes: once this run's own checkpoint exists, switch -t to -r and
#     point at outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_mix/last.pth
#
#   If GPU 0 is busy (e.g. the n_uavdt_LWGA_mix run), change CUDA_VISIBLE_DEVICES.
#   Dependency: the DCT path requires torch_dct (pip install torch_dct).
# ==============================================================================

conda activate deimv2
cd /mnt/e/Experiment/huya-deimv2/deimv2

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1  train.py \
    -c /mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_mix.yml \
    --use-amp --seed=0

python  train.py \
    -c /mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_mix.yml \
    -r /mnt/e/Experiment/huya-deimv2/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_mix/last.pth \
    --use-amp --seed=0

python  train.py \
    -c /mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_mix_1.yml \
    --use-amp --seed=0
