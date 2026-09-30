"""
DEIMv2: Real-Time Object Detection Meets DINOv3
ES-MoE Enhanced HGNetv2 Backbone
---------------------------------------------------------------------------------
This module implements an Efficient Sparse Mixture-of-Experts (ES-MoE) variant
of HGNetv2 backbone, inspired by YOLO-Master.

Key Features:
- Lightweight InvertedResidual experts (2 experts)
- Zero-cost routing using channel statistics (mean + std)
- Shared expert path for training stability
- Load balancing loss + Z-loss for stable training
- Batched expert computation for efficiency

Reference:
- YOLO-Master: https://github.com/Tencent/YOLO-Master
- ES-MoE Paper: arXiv:2512.23273 (CVPR 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import weakref
from .common import FrozenBatchNorm2d, get_activation
from ..core import register

__all__ = ['HGNetv2_MoE']


# =============================================================================
# MoE Loss Registry (from YOLO-Master)
# =============================================================================
MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()


def _get_moe_aux_loss(module: nn.Module) -> torch.Tensor:
    """Retrieve the auxiliary loss from the registry."""
    loss = MOE_LOSS_REGISTRY.get(module)
    if isinstance(loss, torch.Tensor):
        return loss
    # Return zero tensor on same device
    try:
        param = next(module.parameters())
        return param.new_zeros(())
    except StopIteration:
        return torch.tensor(0.0)


# =============================================================================
# Basic Building Blocks (Same as original hgnetv2.py for compatibility)
# =============================================================================

class LearnableAffineBlock(nn.Module):
    """Learnable affine transformation block."""
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            act='relu',
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
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
    """Lightweight Convolution: 1x1 conv + DW conv."""
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
            act='relu',
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            act=act,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            act=act,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    """Stem block for HGNetv2."""
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act='relu'):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            act=act,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            act=act,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            act=act,
        )
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
    """Original ESE module for fallback option."""
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


# =============================================================================
# ES-MoE Core Components (Inspired by YOLO-Master)
# =============================================================================

def get_safe_groups(channels: int, desired_groups: int = 8) -> int:
    """Ensure num_groups divides channels."""
    groups = min(desired_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return max(1, groups)


class InvertedResidualExpert(nn.Module):
    """
    Highly efficient expert module: Uses Inverted Residual structure (MobileNetV2 style).
    2-3x faster than standard convolution experts, fewer parameters, stronger non-linearity.

    Structure:
        1. Pointwise Expand (1x1 Conv)
        2. Depthwise Spatial (3x3 DW Conv)
        3. Pointwise Project (1x1 Conv)
    """
    def __init__(self, in_channels, out_channels, expand_ratio=1, kernel_size=3, act='silu'):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)

        # Use SiLU for better performance (YOLO-Master default)
        activation = nn.SiLU(inplace=True) if act == 'silu' else get_activation(act)

        # 轻量化：expand_ratio=1 时不扩展通道
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # DW Conv only
                nn.Conv2d(in_channels, in_channels, kernel_size,
                          padding=kernel_size // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                activation if act == 'silu' else nn.SiLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                # 1. Pointwise Expand
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation if act == 'silu' else nn.SiLU(inplace=True),
                # 2. Depthwise Spatial
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
                          padding=kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation if act == 'silu' else nn.SiLU(inplace=True),
                # 3. Pointwise Project
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x)


class ZeroCostRouter(nn.Module):
    """
    Zero-cost Router: Reuses feature map statistics for routing decisions.

    Principles (from YOLO-Master):
    1. Uses global average pooling and standard deviation as routing signals
    2. Requires only one Linear layer to map statistics to expert scores
    3. Reduces FLOPs by over 95%
    """
    def __init__(self, in_channels, num_experts, top_k, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = max(float(temperature), 1e-3)

        # Statistics dimension: mean + std = 2 * in_channels
        stat_dim = 2 * in_channels

        # Ultra-lightweight mapping network
        self.router = nn.Sequential(
            nn.Linear(stat_dim, num_experts, bias=False),
            nn.Softmax(dim=1)
        )

        # Initialize with moderate variance for input-dependent routing
        nn.init.normal_(self.router[0].weight, std=0.05)

    def forward(self, x):
        B, C, H, W = x.shape

        # Zero-cost Feature Extraction
        # Global statistics (overlaps with BN computation, near zero cost)
        mean = x.mean(dim=[2, 3])  # [B, C]
        std = x.std(dim=[2, 3], unbiased=False) if H * W > 1 else torch.zeros_like(mean)
        stats = torch.cat([mean, std], dim=1)  # [B, 2C]

        # Routing Decision
        router_logits = self.router(stats) / self.temperature  # [B, num_experts]

        # Clamp for numerical stability
        router_logits = router_logits.clamp(-30.0, 30.0)

        router_probs = F.softmax(router_logits, dim=1)

        # Top-K Selection
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=1)

        # Renormalization
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-6)

        return router_probs, router_logits, topk_weights, topk_indices


class MoELoss(nn.Module):
    """
    Auxiliary losses for MoE models (from YOLO-Master).

    Features:
    - Load balancing loss
    - Z-loss for numerical stability
    - Entropy regularization (optional)
    """
    def __init__(
        self,
        balance_loss_coeff: float = 1.0,
        z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.0,
        num_experts: int = 2,
        top_k: int = 1
    ):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, router_probs, router_logits, topk_indices):
        """
        Args:
            router_probs: [B, num_experts] Full probability distribution
            router_logits: [B, num_experts] Raw logits
            topk_indices: [B, top_k] Selected expert indices
        """
        B = router_probs.shape[0]

        # 1. Load Balancing Loss (GShard style)
        # Importance: mean routing probability per expert
        importance = router_probs.mean(dim=0)  # [num_experts]

        # Usage: fraction of samples routed to each expert
        flat_indices = topk_indices.view(-1)
        local_expert_counts = F.one_hot(flat_indices, num_classes=self.num_experts).float().sum(dim=0)
        usage = local_expert_counts / max(B * self.top_k, 1)
        usage = usage.detach()  # Non-differentiable

        balance_loss = self.num_experts * torch.sum(importance * usage)

        # 2. Z-Loss (Router Stability)
        log_z = torch.logsumexp(router_logits, dim=1)
        z_loss = torch.mean(log_z ** 2)

        # 3. Entropy Loss (optional)
        entropy_loss = torch.tensor(0.0, device=router_probs.device)
        if self.entropy_loss_coeff > 0:
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=1).mean()
            entropy_loss = entropy

        # Total Loss
        total_loss = (self.balance_loss_coeff * balance_loss +
                      self.z_loss_coeff * z_loss +
                      self.entropy_loss_coeff * entropy_loss)

        # NaN Guard
        if not torch.isfinite(total_loss).all():
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return total_loss


class ESMoEModule(nn.Module):
    """
    Efficient Sparse Mixture-of-Experts Module (Inspired by YOLO-Master).

    Key Design:
    1. Zero-cost routing using channel statistics (mean + std)
    2. Lightweight DW Conv experts (2 experts by default)
    3. Shared expert path for training stability
    4. Top-K sparse activation for efficiency
    5. Load balancing loss + Z-loss for stable training

    Args:
        channels: Number of input/output channels
        num_experts: Number of experts (default: 2 for lightweight)
        top_k: Number of experts to activate (default: 1)
        expert_expand_ratio: Expert expansion ratio (default: 1 for ultra-lightweight)
        use_shared_expert: Whether to use shared expert (default: True)
        balance_loss_coeff: Balance loss coefficient (default: 1.0)
        z_loss_coeff: Z-loss coefficient (default: 1.0)
    """
    def __init__(
        self,
        channels,
        num_experts=2,
        top_k=1,
        expert_expand_ratio=1,  # 默认1，超轻量
        use_shared_expert=True,
        balance_loss_coeff=1.0,
        z_loss_coeff=1.0,
        act='relu'
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_shared_expert = use_shared_expert

        # 1. Expert networks (使用 DW Conv，超轻量)
        self.experts = nn.ModuleList([
            InvertedResidualExpert(channels, channels, expand_ratio=expert_expand_ratio, act=act)
            for _ in range(num_experts)
        ])

        # 2. Zero-cost Router
        self.router = ZeroCostRouter(channels, num_experts, self.top_k)

        # 3. Shared Expert (always active for stability) - 也改为DW Conv
        if use_shared_expert:
            self.shared_expert = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.shared_expert = None

        # 4. MoE Loss
        self.moe_loss_fn = MoELoss(
            balance_loss_coeff=balance_loss_coeff,
            z_loss_coeff=z_loss_coeff,
            num_experts=num_experts,
            top_k=self.top_k
        )

        # 5. Gating mechanism (similar to original EseModule)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Routing decision
        router_probs, router_logits, topk_weights, topk_indices = self.router(x)

        # 2. Shared expert forward (always active)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
        else:
            shared_output = 0

        # 3. Sparse expert forward (batched computation)
        expert_output = self._compute_experts_batched(x, topk_weights, topk_indices)

        # 4. Fuse outputs: shared + sparse
        output = shared_output + expert_output

        # 5. Gating (similar to EseModule)
        gate_input = output.mean((2, 3), keepdim=True)
        gate = self.gate(gate_input)
        output = torch.mul(output, gate)

        # 6. Compute auxiliary loss (training only)
        if self.training:
            aux_loss = self.moe_loss_fn(router_probs, router_logits, topk_indices)
            MOE_LOSS_REGISTRY[self] = aux_loss

        return output

    def _compute_experts_batched(self, x, topk_weights, topk_indices):
        """
        Batched expert computation (from YOLO-Master).
        Eliminates for-loops for ~3-5x speedup.
        """
        B, C, H, W = x.shape

        # Initialize output
        expert_output = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)

        # Flatten indices and weights
        indices_flat = topk_indices  # [B, top_k]
        weights_flat = topk_weights  # [B, top_k]

        # Weight threshold for conditional computation
        weight_threshold = 0.01
        valid_mask = weights_flat > weight_threshold

        # Iterate over experts (vectorized over batch)
        for expert_idx in range(self.num_experts):
            # Find all (batch, k) positions that selected this expert
            expert_mask = (indices_flat == expert_idx) & valid_mask

            if not expert_mask.any():
                continue

            batch_indices, k_indices = torch.where(expert_mask)

            # Batched forward pass
            expert_input = x[batch_indices]
            expert_out = self.experts[expert_idx](expert_input)

            # Apply weights
            weights = weights_flat[batch_indices, k_indices].view(-1, 1, 1, 1)
            weighted_out = expert_out * weights

            # Accumulate outputs
            expert_output.index_add_(0, batch_indices, weighted_out.to(expert_output.dtype))

        return expert_output

    @property
    def aux_loss(self):
        """Retrieve the auxiliary loss from the registry."""
        return _get_moe_aux_loss(self)


# =============================================================================
# HG Blocks with MoE
# =============================================================================

class HG_Block_MoE(nn.Module):
    """
    HG Block with ES-MoE attention.

    Similar to HG_Block but uses ESMoEModule instead of EseModule.
    """
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',  # 'ese', 'se', or 'moe'
            drop_path=0.,
            act='relu',
            # MoE specific params
            num_experts=2,
            top_k=1,
            use_shared_expert=True,
            expert_expand_ratio=1,
            balance_loss_coeff=1.0,
            z_loss_coeff=1.0,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        act=act,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                        act=act,
                    )
                )

        # Feature aggregation
        total_chs = in_chs + layer_num * mid_chs

        if agg == 'moe':
            # Use MoE module
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            att = ESMoEModule(
                out_chs,
                num_experts=num_experts,
                top_k=top_k,
                use_shared_expert=use_shared_expert,
                expert_expand_ratio=expert_expand_ratio,
                balance_loss_coeff=balance_loss_coeff,
                z_loss_coeff=z_loss_coeff,
                act=act
            )
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            # Original ESE
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

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


class HG_Stage_MoE(nn.Module):
    """
    HG Stage with ES-MoE blocks.
    """
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='moe',  # Default to MoE
            drop_path=0.,
            act='relu',
            # MoE specific params
            num_experts=2,
            top_k=1,
            use_shared_expert=True,
            expert_expand_ratio=1,
            balance_loss_coeff=1.0,
            z_loss_coeff=1.0,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
                act=act,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block_MoE(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    act=act,
                    num_experts=num_experts,
                    top_k=top_k,
                    use_shared_expert=use_shared_expert,
                    expert_expand_ratio=expert_expand_ratio,
                    balance_loss_coeff=balance_loss_coeff,
                    z_loss_coeff=z_loss_coeff,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# =============================================================================
# HGNetv2 with MoE Backbone
# =============================================================================

@register()
class HGNetv2_MoE(nn.Module):
    """
    HGNetV2 with Efficient Sparse Mixture-of-Experts (ES-MoE).

    This variant of HGNetv2 incorporates MoE modules to achieve
    instance-conditional adaptive computation.

    Key Differences from Original HGNetv2:
    - Uses ESMoEModule instead of EseModule in HG_Blocks
    - 2 lightweight InvertedResidual experts for efficiency
    - Zero-cost routing using channel statistics
    - Shared expert path for training stability
    - Load balancing loss + Z-loss for stable training

    Args:
        name: Architecture name (e.g., 'B0', 'B1', 'B2', etc.)
        use_lab: Whether to use LearnableAffineBlock
        return_idx: Stage indices to return
        freeze_stem_only: Whether to freeze only stem
        freeze_at: Stage index to freeze up to
        freeze_norm: Whether to freeze BatchNorm
        pretrained: Whether to load pretrained weights
        local_model_dir: Directory for pretrained weights
        act: Activation function
        num_experts: Number of MoE experts (default: 2)
        top_k: Number of experts to activate (default: 1)
        use_shared_expert: Whether to use shared expert (default: True)
        moe_stages: Which stages to use MoE (e.g., [2, 3] for stages 2 and 3)
        balance_loss_coeff: Balance loss coefficient (default: 1.0)
        z_loss_coeff: Z-loss coefficient (default: 1.0)
    """

    arch_configs = {
        'Atto': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 256, 1, True, True, 3, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'Femto': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'Pico': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/',
                 act='relu',
                 # MoE specific parameters
                 num_experts=2,
                 top_k=1,
                 use_shared_expert=True,
                 moe_stages=[2, 3],
                 expert_expand_ratio=1,
                 balance_loss_coeff=1.0,
                 z_loss_coeff=1.0,
                 ):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        self.moe_stages = moe_stages
        self.expert_expand_ratio = expert_expand_ratio
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        print(f"        ### HGNetv2_MoE.act: {act} ###")
        print(f"        ### MoE Config: num_experts={num_experts}, top_k={top_k}, moe_stages={moe_stages} ###")

        # Stem
        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
            act=act
        )

        # Stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[k]

            # Determine if this stage uses MoE
            stage_idx = int(k.replace('stage', '')) - 1  # 0-indexed
            use_moe = stage_idx in moe_stages
            agg = 'moe' if use_moe else 'ese'

            self.stages.append(
                HG_Stage_MoE(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    agg=agg,
                    act=act,
                    num_experts=num_experts,
                    top_k=top_k,
                    use_shared_expert=use_shared_expert,
                    expert_expand_ratio=expert_expand_ratio,
                    balance_loss_coeff=balance_loss_coeff,
                    z_loss_coeff=z_loss_coeff,
                )
            )

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
        """Load pretrained weights with partial matching for MoE modules."""
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
                # Try distributed loading
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

            # Load with partial matching (MoE modules will be randomly initialized)
            self._load_partial_state_dict(state, name)

        except Exception as e:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"{str(e)}")
                print(RED + "WARNING: Failed to load pretrained HGNetV2 model. MoE modules will be randomly initialized." + RESET)
            elif not torch.distributed.is_initialized():
                print(f"WARNING: Failed to load pretrained weights: {e}")

    def _load_partial_state_dict(self, state_dict, name):
        """Load state dict with partial matching for MoE compatibility."""
        model_dict = self.state_dict()

        # Filter keys that match in shape
        filtered_dict = {}
        missing_keys = []
        unexpected_keys = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    pass  # Shape mismatch, skip
            else:
                unexpected_keys.append(k)

        # Check for missing keys
        for k in model_dict:
            if k not in filtered_dict:
                missing_keys.append(k)

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)

        # Filter MoE-related keys from missing list
        moe_keywords = ['expert', 'router', 'moe', 'shared_expert']
        moe_missing = [k for k in missing_keys if any(kw in k.lower() for kw in moe_keywords)]
        non_moe_missing = [k for k in missing_keys if k not in moe_missing]

        if non_moe_missing:
            print(f"Missing non-MoE keys ({len(non_moe_missing)}): {non_moe_missing[:5]}...")
        if moe_missing:
            print(f"MoE module keys randomly initialized ({len(moe_missing)} keys)")
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
        """Get auxiliary losses from MoE modules for training."""
        aux_losses = []
        for stage in self.stages:
            for block in stage.blocks:
                if hasattr(block, 'aggregation') and len(block.aggregation) > 1:
                    moe_module = block.aggregation[1]
                    if isinstance(moe_module, ESMoEModule):
                        loss = _get_moe_aux_loss(moe_module)
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


# =============================================================================
# Convenience Functions
# =============================================================================

def build_hgnetv2_moe(
    name='B0',
    num_experts=2,
    top_k=1,
    moe_stages=[2, 3],
    **kwargs
):
    """
    Build HGNetv2 with MoE.

    Args:
        name: Architecture name
        num_experts: Number of experts per MoE module
        top_k: Number of experts to activate
        moe_stages: Which stages to apply MoE
        **kwargs: Additional arguments for HGNetv2_MoE

    Returns:
        HGNetv2_MoE model
    """
    return HGNetv2_MoE(
        name=name,
        num_experts=num_experts,
        top_k=top_k,
        moe_stages=moe_stages,
        **kwargs
    )
