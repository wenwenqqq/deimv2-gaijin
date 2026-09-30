"""HGNetV2-MoE v2 ablation with LWTformer WaveDown at selected stage entries."""

from ..core import register
from .hgnetv2_moe_v2 import HGNetv2_MoE_v2
from ..deim.lwtformer_modules import ChannelPreservingWaveDown

__all__ = ['HGNetv2_MoE_v2_WaveDown']


@register()
class HGNetv2_MoE_v2_WaveDown(HGNetv2_MoE_v2):
    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3],
                 freeze_stem_only=True, freeze_at=0, freeze_norm=True,
                 pretrained=True, local_model_dir='weight/hgnetv2/', act='relu',
                 expert_act='silu', num_experts=3, top_k=2, moe_stages=[2, 3],
                 expert_kernels=[3, 5, 7], expert_expand_ratio=1.0,
                 use_shared_expert=True, balance_loss_coeff=1.0, z_loss_coeff=1.0,
                 wavedown_stages=[2]):
        super().__init__(
            name=name, use_lab=use_lab, return_idx=return_idx,
            freeze_stem_only=freeze_stem_only, freeze_at=freeze_at,
            freeze_norm=freeze_norm, pretrained=pretrained,
            local_model_dir=local_model_dir, act=act, expert_act=expert_act,
            num_experts=num_experts, top_k=top_k, moe_stages=moe_stages,
            expert_kernels=expert_kernels, expert_expand_ratio=expert_expand_ratio,
            use_shared_expert=use_shared_expert,
            balance_loss_coeff=balance_loss_coeff, z_loss_coeff=z_loss_coeff)

        stage_config = self.arch_configs[name]['stage_config']
        for stage_idx in [int(i) for i in wavedown_stages]:
            if stage_idx < 0 or stage_idx >= len(self.stages):
                raise ValueError(f'Invalid WaveDown stage index {stage_idx}.')
            stage_name = list(stage_config.keys())[stage_idx]
            in_channels = int(stage_config[stage_name][0])
            self.stages[stage_idx].downsample = ChannelPreservingWaveDown(in_channels)
