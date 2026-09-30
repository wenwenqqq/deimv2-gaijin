"""
DRFD downsampling module, ported from LWGANet into the DEIM hybrid-encoder codebase.

Source of the module:
    LWGANet (paper: "LWGANet: Addressing Spatial and Channel Redundancy in
    Remote Sensing Visual Tasks with Light-Weight Grouped Attention").
    The paper states (Sec. on architecture / Appendix A, Fig. 4, Table 7):
        "For downsampling between stages, we employ the DRFD module
         (Lu et al. 2023), chosen for its proven ability to preserve fine
         details."  ...  "The DRFD module is responsible for spatial
         downsampling (by a factor of 2) and channel expansion (typically
         doubling)."
    Original implementation lives in
        aaa_test_self/LWGANet/classification/models/LWGANet.py  (class DRFD)
        aaa_test_self/LWGANet/detection/mmrotate/models/backbones/lwganet.py

What this file provides:
    A dependency-free `DRFD` reimplementation that follows the conventions of
    `engine/deim/hybrid_encoder.py` (nn.BatchNorm2d + get_activation), so it can
    be imported / dropped straight into hybrid_encoder.py.

How it maps onto hybrid_encoder.py:
    The current downsampler is `SCDown` (hybrid_encoder.py), stored in
    `self.downsample_convs` and called in the bottom-up PAN path at
        downsample_feat = self.downsample_convs[idx](feat_low)   # hidden_dim -> hidden_dim, s=2
    To replace that slot, build DRFD with out_mult=1 (channel-preserving):
        DRFD(hidden_dim, act=act, out_mult=1)        # hidden_dim -> hidden_dim, H/2
    To reproduce the paper's exact behaviour, use the default out_mult=2:
        DRFD(dim, act=act, out_mult=2)               # dim -> 2*dim, H/2
"""

import os
import sys

import torch
import torch.nn as nn

try:
    from .utils import get_activation
except ImportError:  # allow running this file directly as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import get_activation


class DRFD(nn.Module):
    """DRFD (Detail-pReserving Feature Downsampling) module.

    Spatial:   H, W  ->  H/2, W/2   (stride-2 downsample)
    Channels:  dim   ->  out_mult * dim
        * out_mult=2 (default): paper's channel-doubling  (C  -> 2C)
        * out_mult=1          : channel-preserving drop-in for a slot that
                                must keep channels, e.g. HybridEncoder's
                                `downsample_convs` (SCDown, hidden_dim->hidden_dim).

    Pipeline (input [B, dim, H, W]):
        1) depthwise 3x3 conv (s=1): dim -> out_dim              # channel expand
        2) two parallel stride-2 branches, both -> out_dim at H/2,W/2:
             a) max-pool(3, s=2, p=1)  + norm                   # preserves extrema/edges
             b) depthwise 3x3 conv (s=2) + act + norm            # smooth downsample
        3) concat -> [B, 2*out_dim, H/2, W/2], 1x1 fuse -> out_dim
    """

    def __init__(self, dim, act='silu', out_mult=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert out_mult >= 1, "out_mult must be >= 1"
        out_dim = dim * out_mult
        self.dim = dim
        self.out_dim = out_dim

        # 1) channel expansion (depthwise 3x3, stride 1): dim -> out_dim
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1, groups=dim)

        # 2a) conv branch: depthwise 3x3, stride 2 (downsample) + act + norm
        self.conv_c = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, groups=out_dim)
        self.act_c = get_activation(act)
        self.norm_c = norm_layer(out_dim)

        # 2b) max branch: max pooling stride 2 (downsample) + norm
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = norm_layer(out_dim)

        # 3) fuse the two branches (2*out_dim -> out_dim) with a 1x1 conv
        self.fusion = nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # x: [B, dim, H, W] -> [B, out_dim, H/2, W/2]
        x = self.conv(x)                                # [B, out_dim,     H,   W  ]
        m = self.norm_m(self.max_m(x))                   # [B, out_dim,     H/2, W/2]
        c = self.norm_c(self.act_c(self.conv_c(x)))     # [B, out_dim,     H/2, W/2]
        x = torch.cat([c, m], dim=1)                     # [B, 2*out_dim,   H/2, W/2]
        x = self.fusion(x)                               # [B, out_dim,     H/2, W/2]
        return x


if __name__ == '__main__':
    # Minimal standalone shape check (does not touch hybrid_encoder.py).
    def _out_hw(h):
        # output spatial size for Conv/MaxPool with k=3, s=2, p=1
        return (h + 1) // 2

    def _check(b, c, h, w, out_mult, act):
        m = DRFD(dim=c, act=act, out_mult=out_mult)
        x = torch.randn(b, c, h, w, requires_grad=True)
        y = m(x)
        eh, ew = _out_hw(h), _out_hw(w)
        assert y.shape == (b, c * out_mult, eh, ew), \
            f"out_mult={out_mult}, act={act}: got {tuple(y.shape)}, " \
            f"expected {(b, c * out_mult, eh, ew)}"
        # make sure it is differentiable end-to-end
        y.sum().backward()
        print(f"  [ok] out_mult={out_mult:>2} act={act:<6} "
              f"{tuple(x.shape)} -> {tuple(y.shape)}  (out_dim={c * out_mult})")

    print("DRFD shape check (hybrid_encoder.py conventions: BN + get_activation):")
    _check(2, 256, 16, 16, out_mult=2, act='silu')   # paper-faithful: 256 -> 512, /2
    _check(2, 256, 16, 16, out_mult=1, act='silu')   # drop-in for SCDown: 256 -> 256, /2
    _check(2, 256, 32, 40, out_mult=2, act='gelu')   # rectangular feature map
    _check(1, 128, 13, 13, out_mult=2, act='relu')   # odd / non-power-of-two spatial
    print("all checks passed.")
