"""
HybridEncoder variant whose PAN downsampler is replaced by LWGANet's DRFD module.

This file does NOT modify `hybrid_encoder.py`. It subclasses the original
`HybridEncoder` and swaps only the bottom-up PAN downsampler
(`self.downsample_convs`, originally `SCDown`) for a DRFD module that is
channel-preserving (out_mult=1), so the rest of the encoder — input projection,
transformer, top-down FPN, and the `sum`/`cat` fusion in `forward` — is unchanged.

Why out_mult is fixed to 1 here (channel-preserving):
    `HybridEncoder.forward` fuses the downsampled feature with the next level
    in BOTH branches:
        downsample_feat = self.downsample_convs[idx](feat_low)
        fused_feat = (downsample_feat + feat_height)  if fuse_op=='sum' \
                     else torch.concat([downsample_feat, feat_height], dim=1)
    The FPN/PAN fuse block (`RepNCSPELAN5`) is built with input channels
    `c1 = input_dim`, where `input_dim = hidden_dim` (sum) or `2*hidden_dim`
    (cat). For the PAN concat to yield exactly `2*hidden_dim`, the
    downsampled feature must be `hidden_dim` — the SAME channel count as
    `feat_height`. So regardless of `fuse_op`, this PAN downsampler slot must
    be channel-preserving: hidden_dim -> hidden_dim, stride 2.
    The original `SCDown` it replaces is exactly that (hidden_dim -> hidden_dim,
    stride 2), and DRFD(out_mult=1) matches it 1:1.
    NOTE: the paper's channel-DOUBLING DRFD (dim -> 2*dim) is NOT a drop-in
    here — it would break the PAN fusion channel contract. To use it you'd
    have to also change the PAN fusion math (out of scope for this swap).

Wire-up (config side, for the user to do themselves):
    In a model yml, point the encoder at the new registered name, e.g.:
        DEIM:
          encoder: HybridEncoderDRFD
        HybridEncoderDRFD:
          ...all the same fields as HybridEncoder...
          drfd_act: 'silu'          # activation passed to DRFD

How it relates to the original files:
    * Downsampler being replaced:
        engine/deim/hybrid_encoder.py -> SCDown (lines ~102), stored in
        `self.downsample_convs` (lines ~420-424), used in forward (line ~492).
    * DRFD implementation (ported from LWGANet, dependency-free):
        engine/deim/drfd_downsample.py -> class DRFD
    * Paper source for DRFD:
        aaa_test_self/LWGANet/classification/models/LWGANet.py (class DRFD)
        "For downsampling between stages, we employ the DRFD module
         (Lu et al. 2023) ... spatial downsampling (by a factor of 2) and
         channel expansion (typically doubling)."
"""

import copy

import torch.nn as nn

from .hybrid_encoder import HybridEncoder
from .drfd_downsample import DRFD
from ..core import register

__all__ = ['HybridEncoderDRFD']


@register()
class HybridEncoderDRFD(HybridEncoder):
    """HybridEncoder with the PAN downsampler replaced by DRFD.

    Everything from the parent (`input_proj`, transformer encoder, lateral
    convs, FPN/PAN blocks, position embedding, forward) is inherited as-is.
    Only `self.downsample_convs` is rebuilt with DRFD after the parent builds.
    """

    def __init__(self,
                 drfd_act: str = 'silu',
                 **kwargs):
        # Let the original HybridEncoder build everything (input_proj, encoder,
        # lateral_convs, fpn_blocks, pan_blocks, AND the SCDown downsample_convs).
        super().__init__(**kwargs)

        self.drfd_act = drfd_act
        hidden_dim = self.hidden_dim

        # Rebuild the PAN downsampler list with DRFD (one per PAN level).
        # The parent created `len(in_channels) - 1` SCDown modules; we mirror
        # that count. out_mult is fixed to 1 so DRFD is channel-preserving
        # (hidden_dim -> hidden_dim, stride 2) — a 1:1 drop-in for SCDown, and
        # it satisfies the PAN fusion channel contract for both `sum` and `cat`.
        # (See the module docstring for why out_mult=2 is not usable here.)
        drfd_block = DRFD(
            dim=hidden_dim,
            act=drfd_act,
            out_mult=1,
        )

        self.downsample_convs = nn.ModuleList([
            copy.deepcopy(drfd_block)
            for _ in range(len(self.in_channels) - 1)
        ])

        # forward() is inherited unchanged: it only calls
        #   self.downsample_convs[idx](feat_low)
        # and DRFD(hidden_dim, out_mult=1) is a drop-in for SCDown there.

    def __repr__(self):
        s = super().__repr__()
        return s.rstrip(')') + \
            f"\n  (downsample): DRFD(act='{self.drfd_act}'))"
