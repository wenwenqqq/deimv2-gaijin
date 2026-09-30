"""
DEIMv2 decoder variant with LWGA token self-attention.

LWGANet's LWGA block is a channel-first image operator:
    (B, C, H, W) -> (B, C, H, W)

The DEIM decoder self-attention works on object-query tokens:
    (B, N, C) -> (B, N, C)

This file keeps DEIMv2's decoder contract intact by adapting object-query tokens
to a compact 2D token map, applying an LWGA block, then restoring the original
query order. Cross-attention stays as MSDeformableAttention because it owns the
reference-point sampling from multi-scale encoder memory.

Per-layer control: ``DEIMTransformer_LWGA`` accepts ``use_lwga`` (global default
for every decoder layer) and ``lwga_layers`` (optional explicit list of layer
indices that use LWGA; the remaining layers fall back to the baseline
``nn.MultiheadAttention``). The last decoder layer (``eval_idx``) is the one that
runs at inference, so keeping it on plain MHA often helps final accuracy.
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core import register
from .deim_decoder import DEIMTransformer
from .dfine_decoder import MSDeformableAttention, LQE
from .dfine_utils import weighting_function, distance2bbox
from .deim_utils import RMSNorm, SwiGLUFFN, Gate
from .utils import inverse_sigmoid

__all__ = ['DEIMTransformer_LWGA']


class DropPath(nn.Module):
    """Minimal stochastic-depth layer used by the LWGA block."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


def _make_act(act: str):
    act = str(act).lower()
    if act == 'relu':
        return nn.ReLU
    if act == 'silu':
        return nn.SiLU
    if act == 'gelu':
        return nn.GELU
    raise ValueError(f'Unsupported LWGA activation: {act}')


def _num_groups(dim: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, dim), 0, -1):
        if dim % groups == 0:
            return groups
    return 1


def _build_norm2d(norm_layer, dim: int) -> nn.Module:
    """Small replacement for mmcv.cnn.build_norm_layer used in LWGANet."""

    if norm_layer is None:
        return nn.Identity()
    if isinstance(norm_layer, dict):
        norm_type = str(norm_layer.get('type', 'BN')).lower()
    else:
        norm_type = str(norm_layer).lower()

    if norm_type in ('bn', 'batchnorm', 'batchnorm2d', 'syncbn'):
        return nn.BatchNorm2d(dim)
    if norm_type in ('gn', 'groupnorm'):
        return nn.GroupNorm(_num_groups(dim), dim)
    if norm_type in ('ln', 'layernorm'):
        return nn.GroupNorm(1, dim)
    raise ValueError(f'Unsupported LWGA norm layer: {norm_layer}')


class BlurPool(nn.Module):
    """Dependency-free anti-aliased downsample used by LWGA MRA."""

    def __init__(self, channels: int, stride: int = 3):
        super().__init__()
        filt = torch.tensor([1.0, 2.0, 1.0])
        filt = filt[:, None] * filt[None, :]
        filt = filt / filt.sum()
        self.register_buffer('filt', filt[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.filt.to(dtype=x.dtype), stride=self.stride,
                        padding=1, groups=self.channels)


class PA(nn.Module):
    """Point attention from LWGANet."""

    def __init__(self, dim: int, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            _build_norm2d(norm_layer, dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate_fn(self.p_conv(x))


class LA(nn.Module):
    """Local attention from LWGANet."""

    def __init__(self, dim: int, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            _build_norm2d(norm_layer, dim),
            act_layer(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class MRA(nn.Module):
    """Medium-range attention from LWGANet."""

    def __init__(self, channel: int, att_kernel: int, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1,
                                (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1,
                                (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1,
                                (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1,
                                (1, att_padding), groups=channel, bias=False)
        self.norm = _build_norm2d(norm_layer, channel)

    def forward(self, x: Tensor) -> Tensor:
        x_tem = self.max_m2(self.max_m1(x))
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        # LWGANet's diagonal transforms assume both spatial dimensions stay
        # larger than 1 after anti-aliased pooling. Decoder token chunks can be
        # much smaller, especially DN reconstruction groups, so skip the invalid
        # branch instead of failing on a reshape.
        if min(x_tem.shape[-2:]) > 1:
            x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        else:
            x_h2 = torch.zeros_like(x_h1)
        if min(x_tem.shape[-2:]) > 1:
            x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))
        else:
            x_w2 = torch.zeros_like(x_w1)

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        att = F.interpolate(self.gate_fn(att), size=x.shape[-2:], mode='nearest')
        return x[:, :self.channel] * att

    def h_transform(self, x: Tensor) -> Tensor:
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        return x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)

    def inv_h_transform(self, x: Tensor) -> Tensor:
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        return x[..., 0: shape[-2]]

    def v_transform(self, x: Tensor) -> Tensor:
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x: Tensor) -> Tensor:
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class GA12(nn.Module):
    """LWGANet global branch for early stages."""

    def __init__(self, dim: int, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9,
                                      groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x_, idx = self.downpool(x)
        x_ = self.activation(self.proj_1(x_))
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        sig = self.conv_squeeze(torch.cat([avg_attn, max_attn], dim=1)).sigmoid()
        attn = attn1 * sig[:, 0].unsqueeze(1) + attn2 * sig[:, 1].unsqueeze(1)
        x_ = self.proj_2(x_ * self.conv(attn))
        return self.uppool(x_, indices=idx)


class GA(nn.Module):
    """LWGANet global attention branch."""

    def __init__(self, dim: int, head_dim: int = 4, num_heads: Optional[int] = None,
                 qkv_bias: bool = False, attn_drop: float = 0.,
                 proj_drop: float = 0., proj_bias: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else max(dim // head_dim, 1)
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        n = h * w
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, h, w, self.attention_dim)
        x = self.proj_drop(self.proj(x))
        return x.permute(0, 3, 1, 2)


class D_GA(nn.Module):
    """LWGANet downsampled global branch for stage 2."""

    def __init__(self, dim: int, norm_layer):
        super().__init__()
        self.norm = _build_norm2d(norm_layer, dim)
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x_, idx = self.downpool(x)
        x_ = self.norm(self.attn(x_))
        return self.uppool(x_, indices=idx)


class LWGA_Block(nn.Module):
    """Self-contained port of LWGANet's Light-Weight Grouped Attention block."""

    def __init__(self, dim: int, stage: int = 3, att_kernel: int = 11,
                 mlp_ratio: float = 2., drop_path: float = 0.,
                 act_layer=nn.GELU, norm_layer='GN'):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f'LWGA requires dim divisible by 4, got {dim}.')

        self.stage = int(stage)
        self.dim_split = dim // 4
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            _build_norm2d(norm_layer, mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        )

        self.PA = PA(self.dim_split, norm_layer, act_layer)
        self.LA = LA(self.dim_split, norm_layer, act_layer)
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)
        if self.stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)
        elif self.stage == 3:
            self.GA4 = GA(self.dim_split)
            self.norm = _build_norm2d(norm_layer, self.dim_split)
        else:
            self.GA12 = GA12(self.dim_split, act_layer)
            self.norm = _build_norm2d(norm_layer, self.dim_split)

        self.norm1 = _build_norm2d(norm_layer, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x1, x2, x3, x4 = torch.split(x, self.dim_split, dim=1)
        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)
        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))

        x_att = torch.cat((x1, x2, x3, x4), dim=1)
        return shortcut + self.norm1(self.drop_path(self.mlp(x_att)))


class LWGATokenSelfAttention(nn.Module):
    """Adapter that applies LWGA to decoder query tokens."""

    def __init__(self, dim: int, n_head: int, dropout: float = 0.,
                 stage: int = 3, att_kernel: int = 11, mlp_ratio: float = 2.,
                 drop_path: float = 0., act: str = 'gelu', norm_type: str = 'GN',
                 min_hw: int = 2, sort_by_ref: bool = True,
                 isolate_dn: bool = True):
        super().__init__()
        self.block = LWGA_Block(
            dim=dim,
            stage=stage,
            att_kernel=att_kernel,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            act_layer=_make_act(act),
            norm_layer=norm_type,
        )
        self.fallback_attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)
        self.min_hw = int(min_hw)
        self.sort_by_ref = bool(sort_by_ref)
        self.isolate_dn = bool(isolate_dn)

    def _grid_size(self, n_tokens: int):
        # LWGANet's MRA h/v transforms assume square-ish feature maps. Keep the
        # token map square and pad unused cells, rather than using a tight
        # rectangular packing.
        side = max(self.min_hw, int(math.ceil(math.sqrt(n_tokens))))
        return side, side

    def _order_from_ref(self, reference_points: Optional[Tensor], n_tokens: int):
        if reference_points is None or not self.sort_by_ref:
            return None, None
        ref = reference_points
        if ref.dim() == 4:
            ref = ref[:, :, 0, :]
        xy = ref[:, :n_tokens, :2].detach().clamp(0, 1)
        scores = xy[..., 1] * 1024.0 + xy[..., 0]
        order = torch.argsort(scores, dim=1)
        inv_order = torch.argsort(order, dim=1)
        return order, inv_order

    def _apply_lwga(self, tokens: Tensor, reference_points: Optional[Tensor]) -> Tensor:
        b, n, c = tokens.shape
        if n == 0:
            return tokens

        order, inv_order = self._order_from_ref(reference_points, n)
        if order is not None:
            tokens_sorted = tokens.gather(1, order.unsqueeze(-1).expand(-1, -1, c))
        else:
            tokens_sorted = tokens

        h, w = self._grid_size(n)
        pad = h * w - n
        if pad > 0:
            tokens_map = F.pad(tokens_sorted, (0, 0, 0, pad))
        else:
            tokens_map = tokens_sorted

        feat = tokens_map.transpose(1, 2).reshape(b, c, h, w)
        feat = self.block(feat)
        out_sorted = feat.flatten(2).transpose(1, 2)[:, :n]

        # LWGA_Block has its own residual. Return only the update so the outer
        # decoder residual keeps the same semantics as nn.MultiheadAttention.
        delta_sorted = out_sorted - tokens_sorted
        if inv_order is not None:
            return delta_sorted.gather(1, inv_order.unsqueeze(-1).expand(-1, -1, c))
        return delta_sorted

    def _split_with_dn(self, tokens: Tensor, reference_points: Optional[Tensor], dn_meta):
        dn_split = dn_meta.get('dn_num_split', None) if isinstance(dn_meta, dict) else None
        if not self.isolate_dn or dn_split is None or dn_split[0] == 0:
            return [(tokens, reference_points)]

        dn_tokens, match_tokens = dn_split
        num_group = max(int(dn_meta.get('dn_num_group', 1)), 1)
        group_size = dn_tokens // num_group

        chunks = []
        for i in range(num_group):
            start = i * group_size
            end = dn_tokens if i == num_group - 1 else (i + 1) * group_size
            ref = None if reference_points is None else reference_points[:, start:end]
            chunks.append((tokens[:, start:end], ref))

        ref = None if reference_points is None else reference_points[:, dn_tokens:dn_tokens + match_tokens]
        chunks.append((tokens[:, dn_tokens:dn_tokens + match_tokens], ref))
        return chunks

    def forward(self, target: Tensor, query_pos_embed: Optional[Tensor],
                reference_points: Optional[Tensor], attn_mask=None, dn_meta=None) -> Tensor:
        tokens = target if query_pos_embed is None else target + query_pos_embed

        # Unknown masks cannot be represented by convolutional LWGA. DN masks are
        # handled explicitly by processing reconstruction groups independently.
        if attn_mask is not None and dn_meta is None:
            q = k = tokens
            out, _ = self.fallback_attn(q, k, value=target, attn_mask=attn_mask)
            return out

        chunks = self._split_with_dn(tokens, reference_points, dn_meta)
        if len(chunks) == 1:
            return self._apply_lwga(tokens, reference_points)
        return torch.cat([self._apply_lwga(x, ref) for x, ref in chunks], dim=1)


class MHATokenSelfAttention(nn.Module):
    """Plain-MHA adapter exposing the same 5-arg interface as LWGATokenSelfAttention.

    Lets a decoder layer fall back to the baseline ``nn.MultiheadAttention`` while
    keeping the shared call signature ``(target, query_pos_embed, reference_points,
    attn_mask, dn_meta)`` used by ``TransformerDecoderLayer_LWGA.forward``.
    """

    def __init__(self, dim: int, n_head: int, dropout: float = 0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)

    def forward(self, target: Tensor, query_pos_embed: Optional[Tensor],
                reference_points: Optional[Tensor], attn_mask=None, dn_meta=None) -> Tensor:
        q = k = target if query_pos_embed is None else target + query_pos_embed
        out, _ = self.attn(q, k, value=target, attn_mask=attn_mask)
        return out


class TransformerDecoderLayer_LWGA(nn.Module):
    """DEIM decoder layer with LWGA replacing query self-attention.

    ``use_lwga=False`` swaps the self-attention back to the baseline
    ``nn.MultiheadAttention`` (via :class:`MHATokenSelfAttention`) so a decoder can
    mix LWGA and plain-MHA layers. The forward path is unchanged.
    """

    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 use_gateway=False,
                 use_lwga=True,
                 lwga_stage=3,
                 lwga_att_kernel=11,
                 lwga_mlp_ratio=2.,
                 lwga_drop_path=0.,
                 lwga_act='gelu',
                 lwga_norm_type='GN',
                 lwga_min_hw=2,
                 lwga_sort_by_ref=True,
                 lwga_isolate_dn=True,
                 ):
        super().__init__()

        if layer_scale is not None:
            print(f"     --- Wide Layer@{layer_scale} ---")
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.use_lwga = bool(use_lwga)
        self._attn_meta = (d_model, n_head, dropout)
        self._lwga_kwargs = dict(
            stage=lwga_stage,
            att_kernel=lwga_att_kernel,
            mlp_ratio=lwga_mlp_ratio,
            drop_path=lwga_drop_path,
            act=lwga_act,
            norm_type=lwga_norm_type,
            min_hw=lwga_min_hw,
            sort_by_ref=lwga_sort_by_ref,
            isolate_dn=lwga_isolate_dn,
        )
        self.self_attn = self._build_self_attn(self.use_lwga)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def _build_self_attn(self, use_lwga: bool):
        d_model, n_head, dropout = self._attn_meta
        if use_lwga:
            return LWGATokenSelfAttention(d_model, n_head, dropout=dropout, **self._lwga_kwargs)
        return MHATokenSelfAttention(d_model, n_head, dropout=dropout)

    def set_lwga(self, use_lwga: bool):
        """Swap this layer between LWGA and plain-MHA self-attention.

        Both ops share the same 5-arg call signature, so ``forward`` stays
        unchanged after the swap. Used by ``TransformerDecoder_LWGA`` to build
        per-layer mixed decoders (``lwga_layers``).
        """
        use_lwga = bool(use_lwga)
        if use_lwga != self.use_lwga:
            self.self_attn = self._build_self_attn(use_lwga)
            self.use_lwga = use_lwga

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None,
                dn_meta=None):

        target2 = self.self_attn(target, query_pos_embed, reference_points, attn_mask, dn_meta)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder_LWGA(nn.Module):
    """Transformer decoder copy that passes dn_meta into LWGA self-attention."""

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu', use_lwga=True, lwga_layers=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        # Per-layer self-attention choice:
        #   lwga_layers is not None -> exactly those indices use LWGA (use_lwga ignored).
        #   lwga_layers is None     -> every layer uses LWGA iff use_lwga.
        lwga_set = None if lwga_layers is None else set(int(i) for i in lwga_layers)
        layers = []
        for i in range(num_layers):
            proto = decoder_layer if i <= self.eval_idx else decoder_layer_wide
            layer = copy.deepcopy(proto)
            use = (i in lwga_set) if lwga_set is not None else use_lwga
            layer.set_lwga(use)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act))
                                         for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        project = weighting_function(self.reg_max, up, reg_scale) if not hasattr(self, 'project') else self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes,
                           attn_mask, query_pos_embed, dn_meta=dn_meta)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
            torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores


@register()
class DEIMTransformer_LWGA(DEIMTransformer):
    """DEIMTransformer with LWGA query self-attention in each decoder layer."""

    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_gateway=True,
                 share_bbox_head=False,
                 share_score_head=False,
                 lwga_stage=3,
                 lwga_att_kernel=11,
                 lwga_mlp_ratio=2.,
                 lwga_drop_path=0.,
                 lwga_act='gelu',
                 lwga_norm_type='GN',
                 lwga_min_hw=2,
                 lwga_sort_by_ref=True,
                 lwga_isolate_dn=True,
                 use_lwga=True,
                 lwga_layers=None,
                 ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            mlp_act=mlp_act,
            use_gateway=use_gateway,
            share_bbox_head=share_bbox_head,
            share_score_head=share_score_head,
        )

        decoder_layer = TransformerDecoderLayer_LWGA(
            hidden_dim, nhead, dim_feedforward, dropout,
            activation, num_levels, num_points,
            cross_attn_method=cross_attn_method,
            use_gateway=use_gateway,
            use_lwga=use_lwga,
            lwga_stage=lwga_stage,
            lwga_att_kernel=lwga_att_kernel,
            lwga_mlp_ratio=lwga_mlp_ratio,
            lwga_drop_path=lwga_drop_path,
            lwga_act=lwga_act,
            lwga_norm_type=lwga_norm_type,
            lwga_min_hw=lwga_min_hw,
            lwga_sort_by_ref=lwga_sort_by_ref,
            lwga_isolate_dn=lwga_isolate_dn,
        )
        decoder_layer_wide = TransformerDecoderLayer_LWGA(
            hidden_dim, nhead, dim_feedforward, dropout,
            activation, num_levels, num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
            use_lwga=use_lwga,
            lwga_stage=lwga_stage,
            lwga_att_kernel=lwga_att_kernel,
            lwga_mlp_ratio=lwga_mlp_ratio,
            lwga_drop_path=lwga_drop_path,
            lwga_act=lwga_act,
            lwga_norm_type=lwga_norm_type,
            lwga_min_hw=lwga_min_hw,
            lwga_sort_by_ref=lwga_sort_by_ref,
            lwga_isolate_dn=lwga_isolate_dn,
        )
        self.decoder = TransformerDecoder_LWGA(
            hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
            reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation,
            use_lwga=use_lwga, lwga_layers=lwga_layers)
