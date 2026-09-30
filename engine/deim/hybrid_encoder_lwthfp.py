"""HSFPN encoder ablation replacing its DCT-HFP stack with LWTformer subband gates."""

import torch.nn as nn

from ..core import register
from .hybrid_encoder_HSFPN import HybridEncoder_HSFPN
from .lwtformer_modules import LWTHFPStack

__all__ = ['HybridEncoder_LWTHFP']


@register()
class HybridEncoder_LWTHFP(HybridEncoder_HSFPN):
    __share__ = ['eval_spatial_size']

    def __init__(self,
                 in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32],
                 hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0,
                 enc_act='gelu', use_encoder_idx=[2], num_encoder_layers=1,
                 pe_temperature=10000, expansion=1.0, depth_mult=1.0,
                 act='silu', eval_spatial_size=None, version='dfine',
                 csp_type='csp', fuse_op='cat', hsfpn_depth=2,
                 hsfpn_ratio=[0.25, 0.25], hsfpn_patch=[8, 8],
                 hsfpn_isdct=True, hsfpn_residual=True,
                 hsfpn_out_mode='dw', hsfpn_share_dct=True,
                 lwt_alpha_init=0.1, lwt_directional_init=True):
        super().__init__(
            in_channels=in_channels, feat_strides=feat_strides,
            hidden_dim=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature, expansion=expansion,
            depth_mult=depth_mult, act=act,
            eval_spatial_size=eval_spatial_size, version=version,
            csp_type=csp_type, fuse_op=fuse_op, hsfpn_depth=hsfpn_depth,
            hsfpn_ratio=hsfpn_ratio, hsfpn_patch=hsfpn_patch,
            hsfpn_isdct=hsfpn_isdct, hsfpn_residual=hsfpn_residual,
            hsfpn_out_mode=hsfpn_out_mode, hsfpn_share_dct=hsfpn_share_dct)

        self.encoder = nn.ModuleList([
            LWTHFPStack(
                dim=hidden_dim, depth=hsfpn_depth,
                residual=hsfpn_residual, alpha_init=lwt_alpha_init,
                directional_init=lwt_directional_init,
                out_mode=hsfpn_out_mode)
            for _ in range(len(use_encoder_idx))
        ])
