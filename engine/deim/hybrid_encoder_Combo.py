"""
DEIMv2 HybridEncoder Combo variant: mix HFP and PFG at different feature levels
(异层分工, minimal-redundancy neck stack).

The intra-scale ``TransformerEncoder`` is replaced by a ``ModuleList`` with one
block per entry of ``use_encoder_idx``. Each entry's type is given by the
parallel list ``encoder_types``:
  * 'hfp' -> ``HFPStack`` (from hsfpn_modules): DCT high-pass + channel attention
  * 'pfg' -> a stack of ``PFG`` blocks (from pfg_modules): multi-scale large-kernel
             depthwise + frequency-gated token mixing + GLU channel MLP

Recommended assignment (effect-first, least redundancy):
  use_encoder_idx = [0, 1]   # stride-16, stride-32
  encoder_types   = ['hfp', 'pfg']
  -> HFP on the high-resolution level (tiny-object high-freq, where DCT shines)
  -> PFG on the deep level (large-RF frequency-gated mixing)

Both are channel-first ``(B, C, H, W) -> (B, C, H, W)``, so forward applies them
directly (no flatten / positional embedding). The FPN/PAN fusion path is
unchanged from the original HybridEncoder.

To avoid duplicating the CSP/FPN/PAN helper layers a fourth time, they are
imported from ``.hybrid_encoder``; only the (already self-contained) HFP and PFG
blocks and the combo assembly are defined here.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from .hybrid_encoder import (
    ConvNormLayer_fuse,
    ConvNormLayer,
    SCDown,
    VGGBlock,
    CSPLayer,
    RepNCSPELAN4,
    CSPLayer2,
    RepNCSPELAN5,
)
from .hsfpn_modules import HFPStack
from .pfg_modules import PFG

from ..core import register

__all__ = ['HybridEncoder_Combo']


@register()
class HybridEncoder_Combo(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 csp_type='csp',
                 fuse_op='cat',
                 # combo assignment (one per use_encoder_idx entry)
                 encoder_types=['pfg'],
                 # HFP hyper-parameters (used by 'hfp' entries)
                 hsfpn_depth=2,
                 hsfpn_ratio=[0.25, 0.25],
                 hsfpn_patch=[8, 8],
                 hsfpn_isdct=True,
                 hsfpn_residual=True,
                 # Lightweight-HFP ablation knobs (same as HSFPN)
                 hsfpn_out_mode='conv3x3',
                 hsfpn_share_dct=False,
                 # PFG hyper-parameters (used by 'pfg' entries)
                 pfg_depth=2,
                 pfga_K=[9, 15, 31],
                 center_suppress=False,
                 mlp_ratio=2.0,
                 dw_kernel=3,
                 groups_pw=1,
                 layerscale_init=1e-6,
                 pfg_drop=0.0,
                 pfg_drop_path=0.0,
                 use_grn=False,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.fuse_op = fuse_op

        # combo assignment
        encoder_types = [str(t).lower() for t in encoder_types]
        assert len(encoder_types) == len(use_encoder_idx), \
            "encoder_types must have the same length as use_encoder_idx " \
            f"({len(encoder_types)} vs {len(use_encoder_idx)})"
        for t in encoder_types:
            assert t in ('hfp', 'pfg'), f"encoder_types entry must be 'hfp' or 'pfg', got '{t}'"
        self.encoder_types = encoder_types

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                proj = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ('norm', nn.BatchNorm2d(hidden_dim))
                        ]))
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)

        # ---- intra-scale encoder: per-level HFP / PFG stack (replaces TE) ----
        # nhead/dim_feedforward/enc_act kept for interface compatibility (unused).
        self.hsfpn_depth = int(hsfpn_depth)
        self.hsfpn_ratio = [float(v) for v in hsfpn_ratio]
        self.hsfpn_patch = [int(v) for v in hsfpn_patch]
        self.hsfpn_isdct = bool(hsfpn_isdct)
        self.hsfpn_residual = bool(hsfpn_residual)
        self.hsfpn_out_mode = str(hsfpn_out_mode)
        self.hsfpn_share_dct = bool(hsfpn_share_dct)
        self.pfg_depth = int(pfg_depth)
        self.pfga_K = [int(k) for k in pfga_K]
        self.center_suppress = bool(center_suppress)
        self.mlp_ratio = float(mlp_ratio)
        self.dw_kernel = int(dw_kernel)
        self.groups_pw = int(groups_pw)
        self.layerscale_init = float(layerscale_init)
        self.pfg_drop = float(pfg_drop)
        self.pfg_drop_path = float(pfg_drop_path)
        self.use_grn = bool(use_grn)

        self.encoder = nn.ModuleList()
        for i, _ in enumerate(use_encoder_idx):
            t = self.encoder_types[i]
            if t == 'hfp':
                blk = HFPStack(
                    dim=hidden_dim,
                    depth=self.hsfpn_depth,
                    residual=self.hsfpn_residual,
                    ratio=self.hsfpn_ratio,
                    patch=self.hsfpn_patch,
                    isdct=self.hsfpn_isdct,
                    out_mode=self.hsfpn_out_mode,
                    share_dct=self.hsfpn_share_dct,
                )
            else:  # 'pfg'
                blk = nn.Sequential(*[
                    PFG(
                        dim=hidden_dim,
                        groups_pw=self.groups_pw,
                        layerscale_init=self.layerscale_init,
                        act_layer=nn.GELU,
                        drop=self.pfg_drop,
                        drop_path=(self.pfg_drop_path if j == self.pfg_depth - 1 else 0.0),
                        pfga_K=self.pfga_K,
                        mlp_ratio=self.mlp_ratio,
                        dw_kernel=self.dw_kernel,
                        center_suppress=self.center_suppress,
                        use_grn=self.use_grn,
                    ) for j in range(self.pfg_depth)
                ])
            self.encoder.append(blk)

        input_dim = hidden_dim if self.fuse_op == 'sum' else hidden_dim * 2   # deim use sum instead of cat

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))

        c1, c2, c3, c4, num_blocks = input_dim, hidden_dim, hidden_dim*2, round(expansion * hidden_dim // 2), round(3 * depth_mult)
        if version == 'dfine':
            Fuse_Block = RepNCSPELAN4(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act, csp_type=csp_type)
        elif version == 'deim':
            Fuse_Block = RepNCSPELAN5(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act)
        else:       # RT-DETR
            Fuse_Block = CSPLayer(in_channels=c1, out_channels=c2, num_blocks=num_blocks, act=act, \
                                    expansion=expansion, bottletype=VGGBlock)
            Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act)
            SCDown_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        self._reset_parameters()

    def _reset_parameters(self):
        # HFP/PFG are convolutional and need no positional embedding; kept only
        # for interface parity with the original HybridEncoder.
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder: per-level HFP / PFG, channel-first (no flatten / pos-embed)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                proj_feats[enc_ind] = self.encoder[i](proj_feats[enc_ind])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            fused_feat = (upsample_feat + feat_low) \
                if self.fuse_op == 'sum' else torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](fused_feat)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            fused_feat = (downsample_feat + feat_height) \
                if self.fuse_op == 'sum' else torch.concat([downsample_feat, feat_height], dim=1)
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        return outs
