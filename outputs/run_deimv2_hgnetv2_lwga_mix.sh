#!/bin/bash
# ==============================================================================
# MIXED decoder experiment: LWGA on layers [0,1] + plain MHA on the final
# (eval) layer, i.e. configs/deimv2/deimv2_hgnetv2_n_uavdt_LWGA_mix.yml.
#
# Weights are reused from the all-LWGA run via `-t` (tuning, strict=False):
# backbone/encoder/cross-attn/heads load; only layer-2 self-attn is freshly
# initialized (its state keys differ from LWGA). Use `-r` only if you want to
# resume the mix run's own checkpoint after it produces one.
#
# To compare against the plain all-LWGA baseline (run_deimv2_hgnetv2_lwga.sh),
# everything except the per-layer self-attn choice is identical.
# ==============================================================================

@REM ✅ 默认配置 → [LWGA, LWGA, LWGA]（原行为不变）
@REM ✅ 新配置 → [LWGA, LWGA, MHA]

conda activate deimv2
cd /mnt/e/Experiment/huya-deimv2/deimv2

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1  train.py \
    -c /mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_n_uavdt_LWGA_mix.yml \
    -r /mnt/e/Experiment/huya-deimv2/deimv2/outputs/deimv2_hgnetv2_n_uavdt_LWGA_mix/last.pth \
    --use-amp --seed=0
