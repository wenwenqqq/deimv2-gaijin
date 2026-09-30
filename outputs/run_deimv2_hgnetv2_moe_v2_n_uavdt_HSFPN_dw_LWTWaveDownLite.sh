#!/usr/bin/env bash
# Activate the training environment first. Original experiment scripts unchanged.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${PROJECT_ROOT}/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTWaveDownLite.yml"
OUTPUT="${PROJECT_ROOT}/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTWaveDownLite"
RESUME_ARGS=()
if [ -f "${OUTPUT}/last.pth" ]; then
    RESUME_ARGS=(-r "${OUTPUT}/last.pth")
fi

exec "${PYTHON_BIN:-python}" train.py -c "${CONFIG}" "${RESUME_ARGS[@]}" --use-amp --seed=0 "$@"
