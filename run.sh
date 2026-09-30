CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train.py \
    -c DEIMv2-self-code/configs/deimv2/deimv2_hgnetv2_n_uavdt_TransMixer.yml \
    -r DEIMv2-self-code/outputs/deimv2_hgnetv2_n_uavdt_TransMixer/last.pth \
    --use-amp --seed=0

watch -n 0.5 nvidia-smi

python  train.py \
    -c /root/DEIMv2-self-code/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.yml \
    -r /root/DEIMv2-self-code/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_real_moeloss/last.pth \
    --use-amp --seed=0

python  train.py \
    -c /root/DEIMv2-self-code/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw.yml \
    --use-amp --seed=0

python  train.py \
    -c /root/DEIMv2-self-code/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTSEA_decoder.yml \
    --use-amp --seed=0

python  train.py \
    -c /root/DEIMv2-self-code/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTWaveDownLite.yml \
    --use-amp --seed=0

python  train.py \
    -c /root/DEIMv2-self-code/configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_LWGA.yml \
    -r /root/DEIMv2-self-code/outputs/deimv2_hgnetv2_moe_v2_n_uavdt_LWGA/checkpoint0084.pth \
    --use-amp --seed=0

$env:PYTHONUTF8=1
python train.py -c E:\Experiment\huya-deimv2\deimv2\configs\deimv2\deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTWaveDownLite.yml -r E:\Experiment\huya-deimv2\deimv2\outputs\deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_LWTWaveDownLite\last.pth --use-amp --seed=0

python train.py -c E:\Experiment\huya-deimv2\deimv2\configs\deimv2\deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw.yml -r E:\Experiment\huya-deimv2\deimv2\outputs\deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw\last.pth --output-dir .\outputs\deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_LWGA_dw_250ep --use-amp --seed=0
