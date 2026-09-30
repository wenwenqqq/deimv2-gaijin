"""DEIM decoder ablation with LWTformer SEA value gating on query self-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .deim_decoder import DEIMTransformer, TransformerDecoder
from .dfine_decoder import MSDeformableAttention
from .deim_utils import RMSNorm, SwiGLUFFN, Gate

__all__ = ['DEIMTransformer_LWTSEA']


class LWTSEAQueryAttention(nn.Module):
    """Mask-preserving DEIM MHA plus LWTformer SEA's per-head V gate."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}.')
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.v_gate = nn.Linear(self.head_dim, self.head_dim, bias=True)

    def forward(self, q, k, value, attn_mask=None):
        out, _ = self.attn(q, k, value=value, attn_mask=attn_mask)
        b, n, c = value.shape
        # Reuse MHA's own V projection, then apply SEA's per-head linear/GELU
        # gate.  The MHA path still owns the DN attention mask.
        v_weight = self.attn.in_proj_weight[2 * c:]
        v_bias = None if self.attn.in_proj_bias is None else self.attn.in_proj_bias[2 * c:]
        gate = F.linear(value, v_weight, v_bias)
        gate = gate.reshape(b, n, self.num_heads, self.head_dim)
        gate = F.gelu(self.v_gate(gate))
        out = out.reshape(b, n, self.num_heads, self.head_dim) * gate
        return out.reshape(b, n, c)


class TransformerDecoderLayer_LWTSEA(nn.Module):
    def __init__(self, d_model=256, n_head=8, dim_feedforward=1024,
                 dropout=0., activation='relu', n_levels=4, n_points=4,
                 cross_attn_method='default', layer_scale=None,
                 use_gateway=False):
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = LWTSEAQueryAttention(d_model, n_head, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, target, reference_points, value, spatial_shapes,
                attn_mask=None, query_pos_embed=None):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2 = self.self_attn(q, k, target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points, value, spatial_shapes)
        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = self.norm2(target + self.dropout2(target2))

        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        return self.norm3(target.clamp(min=-65504, max=65504))


@register()
class DEIMTransformer_LWTSEA(DEIMTransformer):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300,
                 feat_channels=[512, 1024, 2048], feat_strides=[8, 16, 32],
                 num_levels=3, num_points=4, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0., activation='relu',
                 num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
                 learn_query_content=False, eval_spatial_size=None, eval_idx=-1,
                 eps=1e-2, aux_loss=True, cross_attn_method='default',
                 query_select_method='default', reg_max=32, reg_scale=4.,
                 layer_scale=1, mlp_act='relu', use_gateway=True,
                 share_bbox_head=False, share_score_head=False):
        super().__init__(
            num_classes=num_classes, hidden_dim=hidden_dim,
            num_queries=num_queries, feat_channels=feat_channels,
            feat_strides=feat_strides, num_levels=num_levels,
            num_points=num_points, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio, box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size, eval_idx=eval_idx, eps=eps,
            aux_loss=aux_loss, cross_attn_method=cross_attn_method,
            query_select_method=query_select_method, reg_max=reg_max,
            reg_scale=reg_scale, layer_scale=layer_scale, mlp_act=mlp_act,
            use_gateway=use_gateway, share_bbox_head=share_bbox_head,
            share_score_head=share_score_head)

        decoder_layer = TransformerDecoderLayer_LWTSEA(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_levels, num_points, cross_attn_method=cross_attn_method,
            use_gateway=use_gateway)
        decoder_layer_wide = TransformerDecoderLayer_LWTSEA(
            hidden_dim, nhead, dim_feedforward, dropout, activation,
            num_levels, num_points, cross_attn_method=cross_attn_method,
            layer_scale=layer_scale, use_gateway=use_gateway)
        self.decoder = TransformerDecoder(
            hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
            reg_max, self.reg_scale, self.up, eval_idx, layer_scale,
            act=activation)
