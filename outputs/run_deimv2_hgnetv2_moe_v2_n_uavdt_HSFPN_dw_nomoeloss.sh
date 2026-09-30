#!/bin/bash
# ==============================================================================
# ABLATION arm C: HSFPN 'dw' + block-level MoE v2 backbone, MoE aux loss DISABLED
#   Single-variable ablation against arm B (the dw run):
#     arm B (baseline): run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.sh
#         -> moe_loss_weight: 0.01   (MoE load-balance + z loss applied)
#         -> uses -r .../HSFPN_dw/last.pth (resumes its own run)
#     arm C (this)    : run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_nomoeloss.sh
#         -> moe_loss_weight: 0        (MoE load-balance + z loss DROPPED)
#         -> trained from scratch (no -r) for a clean single-variable ablation
#   The ONLY config delta vs arm B is `moe_loss_weight` (0.01 -> 0); everything
#   else matches arm B (MoE v2 backbone @ stage4, HFP @ stride-16,
#   hsfpn_out_mode 'dw', vanilla DEIMTransformer decoder, MoE optimizer param
#   groups, sync_bn=False, schedule, aug, matcher). The MoE aux loss is still
#   computed inside each MoEBlock and logged as `moe_aux_loss` for monitoring,
#   but it contributes zero gradient.
#   Trained from scratch (no -r resume). GPU 2,4 / port 29508 (arm B uses 0,1 /
#   29507, so they can run concurrently).
#   Dependency: the DCT path requires `torch_dct` (pip install torch_dct).
#
#   To instead CONTINUE from arm B's already-trained (with MoE loss) checkpoint
#   and only then drop the loss, add the resume flag:
#       -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw/last.pth
#   (that is a different experiment -- not a clean from-scratch ablation).
# ==============================================================================
source /workspace/cpfs-data/miniforge3/etc/profile.d/conda.sh
conda init bash

cd /workspace/cpfs-data/deimv2 && conda activate deimv2
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29508  train.py \
    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_nomoeloss.yml \
    --use-amp --seed=0
