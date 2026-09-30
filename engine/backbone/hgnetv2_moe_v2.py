"""
DEIMv2: HGNetv2 with block-level ES-MoE (v2) -- self-contained.

Option A: replace the ENTIRE HG_Block (not just its SE/ESE attention) in the
selected stages with a `MoEBlock`, whose heterogeneous experts (different
kernels) perform the real in->out transform, so routing yields meaningful
specialization -- faithful to YOLO-Master's ES-MoE (arXiv:2522.23273, CVPR 2026).

This file is a standalone rewrite that REPLACES the HGNetv2 backbone module
(original engine/backbone/hgnetv2.py): it does NOT inherit from, and does NOT
import, hgnetv2_moe.py. Non-MoE stages use the original HG_Block (SE) so they can
still load HGNetv2 pretrained weights; MoE stages are random-initialized.
"""

import os
import logging
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FrozenBatchNorm2d, get_activation
from ..core import register

__all__ = ['HGNetv2_MoE_v2']


# =============================================================================
# MoE auxiliary-loss registry (self-contained, mirrors hgnetv2_moe.py)
# =============================================================================
MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()


def _get_moe_aux_loss(module: nn.Module) -> torch.Tensor:
    loss = MOE_LOSS_REGISTRY.get(module)
    if isinstance(loss, torch.Tensor):
        return loss
    try:
        param = next(module.parameters())
        return param.new_zeros(())
    except StopIteration:
        return torch.tensor(0.0)


# =============================================================================
# Basic building blocks (copied from hgnetv2.py so this file is standalone)
# =============================================================================
class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, groups=1,
                 padding='', use_act=True, use_lab=False, act='relu'):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False)
            )
        else:
            self.conv = nn.Conv2d(
                in_chs, out_chs, kernel_size, stride,
                padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = get_activation(act)
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, groups=1, use_lab=False, act='relu'):
        super().__init__()
        self.conv1 = ConvBNAct(in_chs, out_chs, kernel_size=1, use_act=False, use_lab=use_lab, act=act)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel_size=kernel_size, groups=out_chs,
                               use_act=True, use_lab=use_lab, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act='relu'):
        super().__init__()
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab, act=act)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab, act=act)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act=act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    """Original HG_Block (SE/ESE aggregation). Used for non-MoE stages."""
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3,
                 residual=False, light_block=False, use_lab=False, agg='ese',
                 drop_path=0., act='relu'):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(LightConvBNAct(
                    in_chs if i == 0 else mid_chs, mid_chs,
                    kernel_size=kernel_size, use_lab=use_lab, act=act))
            else:
                self.layers.append(ConvBNAct(
                    in_chs if i == 0 else mid_chs, mid_chs,
                    kernel_size=kernel_size, stride=1, use_lab=use_lab, act=act))

        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(total_chs, out_chs // 2, 1, 1, use_lab=use_lab, act=act)
            aggregation_excitation_conv = ConvBNAct(out_chs // 2, out_chs, 1, 1, use_lab=use_lab, act=act)
            self.aggregation = nn.Sequential(aggregation_squeeze_conv, aggregation_excitation_conv)
        else:
            aggregation_conv = ConvBNAct(total_chs, out_chs, 1, 1, use_lab=use_lab, act=act)
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    """Original HG_Stage. Used for non-MoE stages (agg='se' by default)."""
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3, use_lab=False,
                 agg='se', drop_path=0., act='relu'):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(in_chs, in_chs, kernel_size=3, stride=2,
                                        groups=in_chs, use_act=False, use_lab=use_lab, act=act)
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(HG_Block(
                in_chs if i == 0 else out_chs, mid_chs, out_chs, layer_num,
                residual=False if i == 0 else True, kernel_size=kernel_size,
                light_block=light_block, use_lab=use_lab, agg=agg,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                act=act))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# =============================================================================
# ES-MoE machinery (self-contained)
# =============================================================================
class DynamicRouter(nn.Module):
    """Learned per-sample router (YOLO-Master DynamicRoutingLayer style).

    GAP -> 1x1 conv -> SiLU -> 1x1 conv -> logits -> soft top-k. Compared with a
    zero-cost channel-stat router, the learned 2-layer projection can amplify the
    (small) per-sample variation that survives BatchNorm. Training-time noise on
    the logits forces expert exploration so no expert is starved (post-BN channel
    stats are otherwise near sample-invariant, which collapses routing to a fixed
    ranking and leaves experts dead). Noise is off at inference.
    """
    def __init__(self, in_channels, num_experts, top_k, reduction=8,
                 temperature=1.0, noise_std=1.0):
        super().__init__()
        reduced = max(in_channels // reduction, 8)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.temperature = max(float(temperature), 1e-3)
        self.noise_std = float(noise_std)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.routing_network = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, num_experts, kernel_size=1, bias=False),
        )

    def forward(self, x):
        pooled = self.global_pool(x)                                # (B, C, 1, 1)
        logits = self.routing_network(pooled)                       # (B, E, 1, 1)
        router_logits = logits.view(x.shape[0], self.num_experts)   # (B, E)
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
        router_logits = router_logits.clamp(-30.0, 30.0)
        router_probs = F.softmax(router_logits / self.temperature, dim=1)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=1)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)
        return router_probs, router_logits, topk_weights, topk_indices


class MoELoss(nn.Module):
    def __init__(self, balance_loss_coeff=1.0, z_loss_coeff=1.0,
                 entropy_loss_coeff=0.0, num_experts=2, top_k=1):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, router_probs, router_logits, topk_indices):
        B = router_probs.shape[0]
        importance = router_probs.mean(dim=0)
        flat_indices = topk_indices.view(-1)
        local_expert_counts = F.one_hot(flat_indices, num_classes=self.num_experts).float().sum(dim=0)
        usage = local_expert_counts / max(B * self.top_k, 1)
        usage = usage.detach()
        balance_loss = self.num_experts * torch.sum(importance * usage)
        log_z = torch.logsumexp(router_logits, dim=1)
        z_loss = torch.mean(log_z ** 2)
        entropy_loss = torch.tensor(0.0, device=router_probs.device)
        if self.entropy_loss_coeff > 0:
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=1).mean()
            entropy_loss = entropy
        total_loss = (self.balance_loss_coeff * balance_loss
                      + self.z_loss_coeff * z_loss
                      + self.entropy_loss_coeff * entropy_loss)
        if not torch.isfinite(total_loss).all():
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return total_loss


# =============================================================================
# Heterogeneous expert + block-level MoE
# =============================================================================
class MoEExpert(nn.Module):
    """Lightweight in->out expert: 1x1 expand -> DW kxk -> 1x1 project.
    Kernel `k` provides receptive-field diversity across experts."""
    def __init__(self, in_chs, out_chs, hidden, kernel_size, act='silu'):
        super().__init__()
        self.expand = ConvBNAct(in_chs, hidden, kernel_size=1, use_act=True, use_lab=False, act=act)
        self.dw = ConvBNAct(hidden, hidden, kernel_size=kernel_size, groups=hidden,
                            use_act=True, use_lab=False, act=act)
        self.project = ConvBNAct(hidden, out_chs, kernel_size=1, use_act=False, use_lab=False, act=act)

    def forward(self, x):
        return self.project(self.dw(self.expand(x)))


class MoEBlock(nn.Module):
    """Whole-block MoE replacing an HG_Block. Experts own the in->out transform."""
    def __init__(self, in_chs, out_chs, hidden, num_experts=3, top_k=2,
                 expert_kernels=(3, 5, 7), residual=False, use_shared=True,
                 act='silu', balance_loss_coeff=1.0, z_loss_coeff=1.0):
        super().__init__()
        expert_kernels = list(expert_kernels)
        assert len(expert_kernels) == num_experts, \
            "expert_kernels must have length == num_experts"

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.residual = residual and (in_chs == out_chs)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.experts = nn.ModuleList([
            MoEExpert(in_chs, out_chs, hidden, k, act=act) for k in expert_kernels
        ])
        self.router = DynamicRouter(in_chs, num_experts, self.top_k)
        self.shared_expert = MoEExpert(in_chs, out_chs, hidden, expert_kernels[0], act=act) \
            if use_shared else None
        self.norm = nn.Sequential(nn.BatchNorm2d(out_chs), get_activation(act))

        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff, z_loss_coeff=z_loss_coeff,
            num_experts=num_experts, top_k=self.top_k)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _compute_experts_batched(self, x, topk_weights, topk_indices):
        B, _, H, W = x.shape
        expert_output = torch.zeros(B, self.out_chs, H, W, device=x.device, dtype=x.dtype)
        indices_flat = topk_indices
        weights_flat = topk_weights
        valid_mask = weights_flat > 0.01
        for expert_idx in range(self.num_experts):
            expert_mask = (indices_flat == expert_idx) & valid_mask
            if not expert_mask.any():
                continue
            batch_indices, k_indices = torch.where(expert_mask)
            expert_out = self.experts[expert_idx](x[batch_indices])
            weights = weights_flat[batch_indices, k_indices].view(-1, 1, 1, 1)
            weighted_out = expert_out * weights
            expert_output.index_add_(0, batch_indices, weighted_out.to(expert_output.dtype))
        return expert_output

    @property
    def aux_loss(self):
        return _get_moe_aux_loss(self)

    def forward(self, x):
        router_probs, router_logits, topk_weights, topk_indices = self.router(x)
        shared_output = self.shared_expert(x) if self.shared_expert is not None else 0
        expert_output = self._compute_experts_batched(x, topk_weights, topk_indices)
        output = shared_output + expert_output
        output = self.norm(output)
        if self.residual:
            output = output + x
        if self.training:
            aux_loss = self.moe_loss_fn(router_probs, router_logits, topk_indices)
            MOE_LOSS_REGISTRY[self] = aux_loss
        return output


class HG_Stage_MoE_v2(nn.Module):
    """Stage that uses MoEBlock (whole-block MoE) when `moe=True`, else original HG_Stage."""
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num,
                 downsample=True, light_block=False, kernel_size=3, use_lab=False,
                 act='relu', expert_act='silu', moe=False, num_experts=3, top_k=2,
                 expert_kernels=(3, 5, 7), expert_expand_ratio=1.0,
                 use_shared_expert=True, balance_loss_coeff=1.0, z_loss_coeff=1.0,
                 drop_path=0.):
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(in_chs, in_chs, kernel_size=3, stride=2,
                                        groups=in_chs, use_act=False, use_lab=use_lab, act=act)
        else:
            self.downsample = nn.Identity()

        blocks = []
        for i in range(block_num):
            if moe:
                hidden = int(mid_chs * expert_expand_ratio)
                blocks.append(MoEBlock(
                    in_chs if i == 0 else out_chs, out_chs, hidden,
                    num_experts=num_experts, top_k=top_k,
                    expert_kernels=list(expert_kernels),
                    residual=(i != 0), use_shared=use_shared_expert, act=expert_act,
                    balance_loss_coeff=balance_loss_coeff, z_loss_coeff=z_loss_coeff))
            else:
                blocks.append(HG_Block(
                    in_chs if i == 0 else out_chs, mid_chs, out_chs, layer_num,
                    residual=False if i == 0 else True, kernel_size=kernel_size,
                    light_block=light_block, use_lab=use_lab, agg='se',
                    drop_path=drop_path, act=act))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# =============================================================================
# Backbone
# =============================================================================
@register()
class HGNetv2_MoE_v2(nn.Module):
    """HGNetv2 with block-level ES-MoE (v2), self-contained.

    Stages in `moe_stages` build whole-block `MoEBlock`s (heterogeneous
    in->out experts); other stages keep the original HG_Block so pretrained
    HGNetv2 weights still load for them. MoE aux losses are collected via
    `get_auxiliary_losses()` (det_engine auto-collects when backbone name
    contains 'MoE').
    """

    arch_configs = {
        'Atto': {'stem_channels': [3, 16, 16],
                 'stage_config': {"stage1": [16, 16, 64, 1, False, False, 3, 3],
                                  "stage2": [64, 32, 256, 1, True, False, 3, 3],
                                  "stage3": [256, 64, 256, 1, True, True, 3, 3]},
                 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'},
        'Femto': {'stem_channels': [3, 16, 16],
                  'stage_config': {"stage1": [16, 16, 64, 1, False, False, 3, 3],
                                   "stage2": [64, 32, 256, 1, True, False, 3, 3],
                                   "stage3": [256, 64, 512, 1, True, True, 5, 3]},
                  'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'},
        'Pico': {'stem_channels': [3, 16, 16],
                 'stage_config': {"stage1": [16, 16, 64, 1, False, False, 3, 3],
                                  "stage2": [64, 32, 256, 1, True, False, 3, 3],
                                  "stage3": [256, 64, 512, 2, True, True, 5, 3]},
                 'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'},
        'B0': {'stem_channels': [3, 16, 16],
               'stage_config': {"stage1": [16, 16, 64, 1, False, False, 3, 3],
                                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                                "stage4": [512, 128, 1024, 1, True, True, 5, 3]},
               'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'},
        'B1': {'stem_channels': [3, 24, 32],
               'stage_config': {"stage1": [32, 32, 64, 1, False, False, 3, 3],
                                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                                "stage4": [512, 192, 1024, 1, True, True, 5, 3]},
               'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'},
        'B2': {'stem_channels': [3, 24, 32],
               'stage_config': {"stage1": [32, 32, 96, 1, False, False, 3, 4],
                                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                                "stage4": [768, 256, 1536, 1, True, True, 5, 4]},
               'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'},
        'B3': {'stem_channels': [3, 24, 32],
               'stage_config': {"stage1": [32, 32, 128, 1, False, False, 3, 5],
                                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                                "stage4": [1024, 256, 2048, 1, True, True, 5, 5]},
               'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'},
        'B4': {'stem_channels': [3, 32, 48],
               'stage_config': {"stage1": [48, 48, 128, 1, False, False, 3, 6],
                                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                                "stage4": [1024, 384, 2048, 1, True, True, 5, 6]},
               'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'},
        'B5': {'stem_channels': [3, 32, 64],
               'stage_config': {"stage1": [64, 64, 128, 1, False, False, 3, 6],
                                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                                "stage4": [1024, 512, 2048, 2, True, True, 5, 6]},
               'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'},
        'B6': {'stem_channels': [3, 48, 96],
               'stage_config': {"stage1": [96, 96, 192, 2, False, False, 3, 6],
                                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                                "stage4": [1024, 768, 2048, 3, True, True, 5, 6]},
               'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'},
    }

    def __init__(self, name, use_lab=False, return_idx=[1, 2, 3],
                 freeze_stem_only=True, freeze_at=0, freeze_norm=True,
                 pretrained=True, local_model_dir='weight/hgnetv2/', act='relu',
                 expert_act='silu', num_experts=3, top_k=2, moe_stages=[2, 3],
                 expert_kernels=[3, 5, 7], expert_expand_ratio=1.0,
                 use_shared_expert=True, balance_loss_coeff=1.0, z_loss_coeff=1.0):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.moe_stages = moe_stages
        self.expert_kernels = list(expert_kernels)
        self.expert_expand_ratio = expert_expand_ratio
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.expert_act = expert_act

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        print(f"        ### HGNetv2_MoE_v2.act: {act} (expert_act: {expert_act}) ###")
        print(f"        ### MoE v2 (block-level): num_experts={num_experts}, top_k={top_k}, "
              f"moe_stages={moe_stages}, kernels={self.expert_kernels} ###")

        self.stem = StemBlock(stem_channels[0], stem_channels[1], stem_channels[2],
                              use_lab=use_lab, act=act)

        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]
            stage_idx = int(k.replace('stage', '')) - 1
            moe = stage_idx in moe_stages
            self.stages.append(HG_Stage_MoE_v2(
                in_channels, mid_channels, out_channels, block_num, layer_num,
                downsample, light_block, kernel_size, use_lab, act=act, expert_act=expert_act, moe=moe,
                num_experts=num_experts, top_k=top_k, expert_kernels=self.expert_kernels,
                expert_expand_ratio=expert_expand_ratio, use_shared_expert=use_shared_expert,
                balance_loss_coeff=balance_loss_coeff, z_loss_coeff=z_loss_coeff))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            self._load_pretrained(name, download_url, local_model_dir)

    def _load_pretrained(self, name, download_url, local_model_dir):
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
        try:
            if name in ['Atto', 'Femto', 'Pico']:
                model_path = local_model_dir + 'PPHGNetV2_B0_stage1.pth'
            else:
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
            if os.path.exists(model_path):
                state = torch.load(model_path, map_location='cpu')
                print(f"Loaded stage1 {name} HGNetV2 from local file.")
            else:
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(GREEN + "Downloading HGNetV2 pretrained weights..." + RESET)
                    state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                    torch.distributed.barrier()
                elif torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    state = torch.load(local_model_dir)
                else:
                    print("Downloading HGNetV2 pretrained weights...")
                    state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                print(f"Loaded stage1 {name} HGNetV2 from URL.")
            # block-level MoE stages won't match pretrained -> partial load
            self._load_partial_state_dict(state)
        except Exception as e:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"{str(e)}")
                print(RED + "WARNING: Failed to load pretrained HGNetV2. MoE stages will be randomly initialized." + RESET)
            elif not torch.distributed.is_initialized():
                print(f"WARNING: Failed to load pretrained weights: {e}")

    def _load_partial_state_dict(self, state_dict):
        model_dict = self.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}
        missing_keys = [k for k in model_dict if k not in filtered_dict]
        unexpected_keys = [k for k in state_dict if k not in filtered_dict]
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)
        moe_keywords = ['expert', 'router', 'moe', 'shared_expert']
        moe_missing = [k for k in missing_keys if any(kw in k.lower() for kw in moe_keywords)]
        non_moe_missing = [k for k in missing_keys if k not in moe_missing]
        if non_moe_missing:
            print(f"Missing non-MoE keys ({len(non_moe_missing)}): {non_moe_missing[:5]}...")
        if moe_missing:
            print(f"MoE v2 block keys randomly initialized ({len(moe_missing)} keys)")
        if unexpected_keys:
            print(f"Unexpected keys in pretrained ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def get_auxiliary_losses(self):
        aux_losses = []
        for stage in self.stages:
            for block in stage.blocks:
                if isinstance(block, MoEBlock):
                    loss = _get_moe_aux_loss(block)
                    if isinstance(loss, torch.Tensor) and loss.requires_grad:
                        aux_losses.append(loss)
        return aux_losses

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
