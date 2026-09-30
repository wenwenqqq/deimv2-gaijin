#!/bin/bash
# LWTformer experiment 1: learnable WaveDown at HGNetV2 stride-8 -> stride-16.

cd /workspace/cpfs-data/deimv2 && conda activate deimv2

CONFIG=/mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTWaveDown.yml
OUTPUT=/mnt/e/Experiment/huya-deimv2/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTWaveDown
RESUME_ARGS=()
if [ -f "${OUTPUT}/last.pth" ]; then
    RESUME_ARGS=(-r "${OUTPUT}/last.pth")
fi

python train.py -c "${CONFIG}" "${RESUME_ARGS[@]}" --use-amp --seed=0
