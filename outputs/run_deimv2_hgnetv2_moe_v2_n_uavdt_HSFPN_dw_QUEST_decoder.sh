#!/usr/bin/env bash
# Activate your deimv2 environment before running this script.
# QUEST in every decoder query self-attention; resume only this experiment.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CONFIG="${ROOT}/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_decoder.yml"
OUTPUT="${ROOT}/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_decoder"
RESUME_ARGS=()
if [ -f "${OUTPUT}/last.pth" ]; then
    RESUME_ARGS=(-r "${OUTPUT}/last.pth")
fi

"${PYTHON_BIN:-python}" train.py -c "${CONFIG}" "${RESUME_ARGS[@]}" --use-amp --seed=0 "$@"


conda activate deimv2
cd E:\Experiment\huya-deimv2\deimv2

python train.py -c configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_decoder.yml  --use-amp --seed=0

python train.py   -c configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_decoder.yml  -r outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_decoder/last.pth  --use-amp --seed=0