#!/bin/bash
# ==============================================================================
# ABLATION arm B: HSFPN 'dw' + block-level MoE v2 backbone
#   Single-variable ablation against the sibling MoE v2 + HSFPN run:
#     arm A (baseline): run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN.sh -> hsfpn_out_mode 'dw_pw'
#     arm B (this)    : run_deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.sh -> hsfpn_out_mode 'dw'
#   The ONLY delta is the HFP output projection; everything else matches arm A
#   (MoE v2 backbone @ stage4, HFP @ stride-16, vanilla DEIMTransformer decoder,
#   MoE optimizer param groups, sync_bn=False, moe_loss_weight).
#   Trained from scratch (no -r resume). GPU 6,7 / port 29507 (arm A uses 2,3 /
#   29501, so they can run concurrently).
#   Dependency: the DCT path requires `torch_dct` (pip install torch_dct).
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29507  train.py \
#    -c /workspace/cpfs-data/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.yml \
#    -r /workspace/cpfs-data/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw/last.pth \
#    --use-amp --seed=0
# ==============================================================================


conda activate deimv2
cd /mnt/e/Experiment/huya-deimv2/deimv2

#python  train.py \
#    -c /mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.yml \
#    -r /mnt/e/Experiment/huya-deimv2/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_real_moeloss/last.pth \
#    --use-amp --seed=0

python  train.py \
    -c /mnt/e/Experiment/huya-deimv2/deimv2/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.yml \
    -r /mnt/e/Experiment/huya-deimv2/deimv2/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_real_moeloss/last.pth \
    --use-amp --seed=0