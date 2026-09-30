#!/bin/bash
# LWTformer experiment 2: learnable directional subband HFP at stride-16.

cd /workspace/cpfs-data/deimv2 && conda activate deimv2

CONFIG=/workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTHFP.yml
OUTPUT=/workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTHFP
RESUME_ARGS=()
if [ -f "${OUTPUT}/last.pth" ]; then
    RESUME_ARGS=(-r "${OUTPUT}/last.pth")
fi

python train.py -c "${CONFIG}" "${RESUME_ARGS[@]}" --use-amp --seed=0
