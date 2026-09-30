"""
HS-FPN modules, ported from HS-FPN (AAAI 2025).

Source:
    HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection
    https://arxiv.org/abs/2412.10116

This file is a self-contained port of the original ``hs_fpn.py`` HFP module
(``DctSpatialInteraction`` / ``DctChannelInteraction`` / ``HFP``), with all
MMDetection / mmcv dependencies stripped:

  * ``mmcv.cnn.ConvModule`` -> plain ``nn.Conv2d`` (+ optional norm/act)
  * ``mmcv.runner.BaseModule`` -> ``nn.Module`` (init_cfg dropped; Xavier init
    applied directly to Conv2d to match the original intent)
  * ``auto_fp16`` -> removed; AMP is handled externally via ``--use-amp``. The
    DCT ops are forced to fp32 inside the forward to avoid fp16 DCT kernels.

Only the spatial ``(B, C, H, W) -> (B, C, H, W)`` HFP block is kept; the SDP
cross-attention and the full 4-level ``HS_FPN`` neck are intentionally not
ported (HFP is used here as an intra-scale enhancement block inside the
DEIMv2 HybridEncoder).

Dependency note:
    ``torch_dct`` is imported lazily so that simply importing this module (and
    thus the whole ``engine.deim`` package) never fails when ``torch_dct`` is
    absent. The DCT path is only exercised when ``isdct=True``; if ``torch_dct``
    is missing at that point a clear RuntimeError is raised pointing to
    ``pip install torch_dct``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_dct as DCT
    _HAS_TORCH_DCT = True
except ImportError:  # pragma: no cover
    DCT = None
    _HAS_TORCH_DCT = False


def _require_dct():
    if not _HAS_TORCH_DCT:
        raise RuntimeError(
            "HS-FPN HFP needs the 'torch_dct' package for the DCT path "
            "(isdct=True). Install it with:  pip install torch_dct"
        )


def _safe_groups(channels: int, desired: int = 32) -> int:
    """num_groups that divides channels (for GroupNorm and grouped Conv2d)."""
    g = min(desired, channels)
    while channels % g != 0:
        g -= 1
    return max(1, g)


# ------------------------------------------------------------------ #
# Output projection of HFP.
# The original HFP uses a full 3x3 conv here, which is ~100% of the
# block's params/MACs (the DCT high-pass masks are parameter-free). The
# HS-FPN *idea* (DCT high-pass + spatial/channel masking) is independent
# of this projection, so it is the natural simplification target. The
# modes below let config files ablate the cost/accuracy trade-off without
# touching the high-frequency perception path:
#   'conv3x3' : original 3x3 conv + GN                        (baseline)
#   'dw_pw'   : depthwise 3x3 + pointwise 1x1 + GN  (factored)  ~8x cheaper
#   'dw'      : depthwise 3x3 only + GN                       ~50x cheaper
#   '1x1'     : 1x1 channel mix + GN                          ~8x cheaper
#   'none'    : Identity (pure masking; residual carries signal) ~0 cost
# ------------------------------------------------------------------ #
def _build_out(dim: int, out_mode: str = 'conv3x3') -> nn.Module:
    g = _safe_groups(dim, 32)
    if out_mode == 'conv3x3':
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(g, dim),
        )
    if out_mode == 'dw_pw':
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.GroupNorm(g, dim),
        )
    if out_mode == 'dw':
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.GroupNorm(g, dim),
        )
    if out_mode == '1x1':
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.GroupNorm(g, dim),
        )
    if out_mode == 'none':
        return nn.Identity()
    raise ValueError(
        f"unknown hsfpn_out_mode={out_mode!r}; expected one of "
        f"['conv3x3', 'dw_pw', 'dw', '1x1', 'none']"
    )


# ------------------------------------------------------------------ #
# Spatial Path of HFP (only p1&p2 use DCT in the original)
# ------------------------------------------------------------------ #
class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio, isdct=True):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct
        if not self.isdct:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def highpass(self, x):
        """DCT high-frequency response (idct of the high-passed spectrum).

        Returned in fp32 (the DCT runs in fp32 for AMP safety); callers cast
        to the input dtype where needed. Factored out so the spatial and
        channel paths can *share* one DCT/IDCT pair instead of each computing
        it (the two masks are identical, so sharing is exact, not an approx).
        """
        _, _, h0, w0 = x.size()
        _require_dct()
        xf = x.float()                                     # DCT in fp32 (AMP-safe)
        idct = DCT.dct_2d(xf, norm='ortho')
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)
        dct = idct * weight                                # filter out low-freq
        return DCT.idct_2d(dct, norm='ortho')              # spatial high-freq mask (fp32)

    def forward(self, x, hp=None):
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        if hp is None:
            hp = self.highpass(x)
        return x * hp.to(x.dtype)

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------ #
# Channel Path of HFP (only p1&p2 use DCT in the original)
# ------------------------------------------------------------------ #
class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch, ratio, isdct=True):
        super().__init__()
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        g = _safe_groups(in_channels, 32)
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=g, bias=False)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=g, bias=False)
        self.relu = nn.ReLU()

    def highpass(self, x):
        """DCT high-frequency response, fp32 (shared with the spatial path)."""
        n, c, h, w = x.size()
        _require_dct()
        xf = x.float()
        idct = DCT.dct_2d(xf, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight
        return DCT.idct_2d(dct, norm='ortho')

    def forward(self, x, hp=None):
        n, c, h, w = x.size()
        if not self.isdct:
            # NOTE: output_size passed positionally so calflops' patched
            # _pool_flops_compute (which has no `output_size` kwarg) does not
            # crash during the FLOPs-profiling forward. Semantics are identical.
            amaxp = F.adaptive_max_pool2d(x, (1, 1))
            aavgp = F.adaptive_avg_pool2d(x, (1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))

        if hp is None:
            hp = self.highpass(x)
        amaxp = F.adaptive_max_pool2d(hp, (self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(hp, (self.h, self.w))
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel)).to(x.dtype)

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


# ------------------------------------------------------------------ #
# High Frequency Perception Module HFP
# ------------------------------------------------------------------ #
class HFP(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True,
                 out_mode='conv3x3', share_dct=False):
        super().__init__()
        # defensive casts: YAML may deliver scalars as strings / tuples as lists
        in_channels = int(in_channels)
        ratio = (float(ratio[0]), float(ratio[1]))
        patch = (int(patch[0]), int(patch[1]))
        isdct = bool(isdct)
        self.out_mode = str(out_mode)
        # share_dct only matters on the DCT path; on the plain-attention path
        # (isdct=False) there is no DCT to share, so it is forced off.
        self.share_dct = bool(share_dct) and isdct

        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = _build_out(in_channels, self.out_mode)
        self._init_params()

    def _init_params(self):
        # original used Xavier-uniform on Conv2d (init_cfg)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.share_dct:
            # one DCT/IDCT pair, shared by both paths (exact, ~halves DCT cost)
            hp = self.spatial.highpass(x)
            spatial = self.spatial(x, hp)
            channel = self.channel(x, hp)
        else:
            spatial = self.spatial(x)
            channel = self.channel(x)
        return self.out(spatial + channel)


# ------------------------------------------------------------------ #
# Stack of HFP blocks used as the intra-scale encoder.
# A residual connection is added around each block (the single original HFP has
# none) so that depth>1 stacks remain stable and information-preserving.
# ------------------------------------------------------------------ #
class HFPStack(nn.Module):
    def __init__(self, dim, depth=2, residual=True,
                 ratio=(0.25, 0.25), patch=(8, 8), isdct=True,
                 out_mode='conv3x3', share_dct=False):
        super().__init__()
        dim = int(dim)
        depth = int(depth)
        self.residual = bool(residual)
        self.blocks = nn.ModuleList([
            HFP(dim, ratio=ratio, patch=patch, isdct=isdct,
                out_mode=out_mode, share_dct=share_dct) for _ in range(depth)
        ])

    def forward(self, x):
        for b in self.blocks:
            y = b(x)
            x = x + y if self.residual else y
        return x
