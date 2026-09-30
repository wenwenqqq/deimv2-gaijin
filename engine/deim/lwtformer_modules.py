"""Detection-oriented ports of the reusable LWTformer wavelet components.

The modules in this file are additive variants for ablation experiments.  They
do not alter the existing HSFPN or HGNetV2 implementations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, limit: int = 32) -> int:
    for groups in range(min(channels, limit), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class LearnableHaarDWT2D(nn.Module):
    """One-level, channel-wise 2D DWT with learnable Haar-initialized filters."""

    def __init__(self):
        super().__init__()
        scale = 1.0 / math.sqrt(2.0)
        self.dec_lo = nn.Parameter(torch.tensor([scale, scale], dtype=torch.float32))
        self.dec_hi = nn.Parameter(torch.tensor([-scale, scale], dtype=torch.float32))

    def forward(self, x):
        channels = x.shape[1]
        lo = self.dec_lo.to(dtype=x.dtype)
        hi = self.dec_hi.to(dtype=x.dtype)
        kernels = torch.stack([
            torch.outer(lo, lo),  # LL: approximation
            torch.outer(hi, lo),  # LH: horizontal detail
            torch.outer(lo, hi),  # HL: vertical detail
            torch.outer(hi, hi),  # HH: diagonal detail
        ], dim=0)
        weight = kernels[:, None].repeat(channels, 1, 1, 1)
        bands = F.conv2d(x, weight, stride=2, groups=channels)
        bands = bands.view(x.shape[0], channels, 4, bands.shape[-2], bands.shape[-1])
        return bands[:, :, 0], bands[:, :, 1], bands[:, :, 2], bands[:, :, 3]


class DirectionalWaveletGate(nn.Module):
    """LWTformer-style LL/LH/HL/HH refinement followed by a channel-spatial gate."""

    def __init__(self, dim: int, directional_init: bool = True):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f'DirectionalWaveletGate requires dim divisible by 4, got {dim}.')
        branch_dim = dim // 4
        self.dwt = LearnableHaarDWT2D()

        self.ll = nn.Sequential(
            nn.Conv2d(dim, branch_dim, 1, bias=False),
            nn.Conv2d(branch_dim, branch_dim, 3, padding=1, groups=branch_dim, bias=False),
        )
        self.lh_proj = nn.Conv2d(dim, branch_dim, 1, bias=False)
        self.lh_context = nn.Conv2d(branch_dim, branch_dim, (1, 3), padding=(0, 1),
                                    groups=branch_dim, bias=False)
        self.hl_proj = nn.Conv2d(dim, branch_dim, 1, bias=False)
        self.hl_context = nn.Conv2d(branch_dim, branch_dim, (3, 1), padding=(1, 0),
                                    groups=branch_dim, bias=False)
        self.hh_proj = nn.Conv2d(dim, branch_dim, 1, bias=False)
        self.hh_act = nn.Tanh()

        self.lh_direction = nn.Conv2d(branch_dim, branch_dim, 3, padding=1,
                                      groups=branch_dim, bias=False)
        self.hl_direction = nn.Conv2d(branch_dim, branch_dim, 3, padding=1,
                                      groups=branch_dim, bias=False)
        self.hh_direction = nn.Conv2d(branch_dim, branch_dim, 3, padding=1,
                                      groups=branch_dim, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        if directional_init:
            self._init_directional_priors()

    @torch.no_grad()
    def _init_directional_priors(self):
        horizontal = torch.tensor([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]])
        vertical = torch.tensor([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
        diagonal = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        for conv, kernel in ((self.lh_direction, horizontal),
                             (self.hl_direction, vertical),
                             (self.hh_direction, diagonal)):
            conv.weight.copy_(kernel.view(1, 1, 3, 3).repeat(conv.out_channels, 1, 1, 1))

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        ll = self.ll(ll)
        lh = self.lh_direction(self.lh_context(self.lh_proj(lh)))
        hl = self.hl_direction(self.hl_context(self.hl_proj(hl)))
        hh = self.hh_direction(self.hh_proj(self.hh_act(hh)))
        gate = self.fuse(torch.cat([ll, lh, hl, hh], dim=1))
        return F.interpolate(gate, size=x.shape[-2:], mode='bilinear', align_corners=False)


class WaveletDownGate(nn.Module):
    """The WaveDown paper path: four subbands, DWConv compression and sigmoid."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwt = LearnableHaarDWT2D()
        bands = dim * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(bands, bands, 3, padding=1, groups=bands, bias=False),
            nn.GroupNorm(_groups(bands), bands),
            nn.GELU(),
            nn.Conv2d(bands, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        return self.fuse(torch.cat([ll, lh, hl, hh], dim=1))


class ChannelPreservingWaveDown(nn.Module):
    """WaveDown adapted to preserve channels at an HGNetV2 stage boundary."""

    def __init__(self, channels: int):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(f'WaveDown requires channels divisible by 4, got {channels}.')
        self.main = nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False)
        self.wavelet_gate = WaveletDownGate(channels)

    def forward(self, x):
        main = F.pixel_unshuffle(self.main(x), 2)
        gate = self.wavelet_gate(x)
        return main * gate + main


class LWTHFP(nn.Module):
    """LWTformer directional subband gate with the baseline HSFPN-dw projection."""

    def __init__(self, dim: int, alpha_init: float = 0.1, directional_init: bool = True,
                 out_mode: str = 'dw'):
        super().__init__()
        if out_mode != 'dw':
            raise ValueError("The LWTHFP ablation intentionally supports only out_mode='dw'.")
        self.gate = DirectionalWaveletGate(dim, directional_init=directional_init)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True),
            nn.GroupNorm(_groups(dim), dim),
        )

    def forward(self, x):
        return self.out(x * (1.0 + self.alpha * self.gate(x)))


class LWTHFPStack(nn.Module):
    def __init__(self, dim: int, depth: int = 1, residual: bool = True,
                 alpha_init: float = 0.1, directional_init: bool = True,
                 out_mode: str = 'dw'):
        super().__init__()
        self.residual = bool(residual)
        self.blocks = nn.ModuleList([
            LWTHFP(dim, alpha_init=alpha_init, directional_init=directional_init,
                   out_mode=out_mode) for _ in range(int(depth))
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x) if self.residual else block(x)
        return x
