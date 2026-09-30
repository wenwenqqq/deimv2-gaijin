"""Independent HGNetV2-MoE experiment with lightweight WaveDown stage entries."""

from ..core import register
from .hgnetv2_moe_v2 import HGNetv2_MoE_v2
from ..deim.lwtformer_modules_lite import ChannelPreservingWaveDownLite

__all__ = ['HGNetv2_MoE_v2_WaveDownLite']


@register()
class HGNetv2_MoE_v2_WaveDownLite(HGNetv2_MoE_v2):
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

        stage_names = list(self.arch_configs[name]['stage_config'])
        for stage_idx in [int(i) for i in wavedown_stages]:
            if stage_idx <= 0 or stage_idx >= len(self.stages):
                raise ValueError(f'WaveDownLite requires a downsampling stage, got {stage_idx}.')
            channels = int(self.arch_configs[name]['stage_config'][stage_names[stage_idx]][0])
            downsample = ChannelPreservingWaveDownLite(channels)
            if freeze_norm:
                downsample = self._freeze_norm(downsample)
            if freeze_at >= stage_idx and not freeze_stem_only:
                self._freeze_parameters(downsample)
            self.stages[stage_idx].downsample = downsample
