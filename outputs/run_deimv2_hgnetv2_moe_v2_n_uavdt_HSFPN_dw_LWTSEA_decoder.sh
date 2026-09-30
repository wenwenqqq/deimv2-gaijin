#!/bin/bash
# LWTformer experiment 3: SEA Value gate in the DEIM decoder self-attention.

# conda activate deimv2
# cd /mnt/e/Experiment/huya-deimv2/deimv2

CONFIG=/mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTSEA_decoder.yml
OUTPUT=/mnt/e/Experiment/huya-deimv2/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTSEA_decoder
RESUME_ARGS=()
if [ -f "${OUTPUT}/last.pth" ]; then
    RESUME_ARGS=(-r "${OUTPUT}/last.pth")
fi

python train.py -c "${CONFIG}" "${RESUME_ARGS[@]}" --use-amp --seed=0
