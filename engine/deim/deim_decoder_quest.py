"""QUEST on DEIM query self-attention (https://arxiv.org/abs/2604.00199).

Only projected keys are L2-normalized per head; attention scale is exactly 1.
The inherited decoder retains its deformable cross-attention, FFN and DN path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .deim_decoder import DEIMTransformer

__all__ = ['DEIMTransformer_QUEST']


class QUESTQueryAttention(nn.MultiheadAttention):
    """Batch-first DEIM attention with MHA-compatible parameter names/shapes.

This internal adapter returns no attention weights, as DEIM discards them.
Boolean masks follow MHA semantics: True blocks attention (opposite to SDPA).
"""

    def forward(self, query, key, value, attn_mask=None):
        b, nq, c = query.shape
        nk = key.shape[1]
        weights = self.in_proj_weight.chunk(3, dim=0)
        biases = (None,) * 3 if self.in_proj_bias is None else self.in_proj_bias.chunk(3)
        q, k, v = [F.linear(x, w, bias).reshape(b, -1, self.num_heads, self.head_dim)
                   .transpose(1, 2)
                   for x, w, bias in zip((query, key, value), weights, biases)]

        # FP32 normalization avoids epsilon underflow for zero keys under AMP.
        norm_dtype = torch.float32 if k.dtype in (torch.float16, torch.bfloat16) else k.dtype
        k = F.normalize(k.to(norm_dtype), p=2, dim=-1, eps=1e-6).to(q.dtype)

        mask = attn_mask
        if mask is not None:
            if mask.ndim == 2:
                if mask.shape != (nq, nk):
                    raise ValueError('Expected a [query_length, key_length] attention mask.')
                mask = mask[None, None]
            elif mask.ndim == 3:
                if mask.shape != (b * self.num_heads, nq, nk):
                    raise ValueError('Expected a [batch * heads, query_length, key_length] mask.')
                mask = mask.reshape(b, self.num_heads, nq, nk)
            else:
                raise ValueError('DEIM attention masks must have 2 or 3 dimensions.')
            mask = ~mask if mask.dtype == torch.bool else mask.to(q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0, scale=1.0)
        out = out.transpose(1, 2).reshape(b, nq, c)
        return self.out_proj(out), None


@register()
class DEIMTransformer_QUEST(DEIMTransformer):
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
            num_queries=num_queries, feat_channels=feat_channels, feat_strides=feat_strides,
            num_levels=num_levels, num_points=num_points, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            num_denoising=num_denoising, label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale, learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size, eval_idx=eval_idx, eps=eps,
            aux_loss=aux_loss, cross_attn_method=cross_attn_method,
            query_select_method=query_select_method, reg_max=reg_max, reg_scale=reg_scale,
            layer_scale=layer_scale, mlp_act=mlp_act, use_gateway=use_gateway,
            share_bbox_head=share_bbox_head, share_score_head=share_score_head)

        # Preserve initialization and all non-attention modules exactly. Avoid
        # consuming extra RNG values when creating replacement projections.
        with torch.random.fork_rng(devices=[]):
            for layer in self.decoder.layers:
                original = layer.self_attn
                attention = QUESTQueryAttention(
                    original.embed_dim, original.num_heads,
                    dropout=original.dropout, batch_first=True)
                attention.load_state_dict(original.state_dict(), strict=True)
                layer.self_attn = attention
