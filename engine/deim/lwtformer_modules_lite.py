"""Lightweight WaveDown variant; the original LWT modules remain unchanged."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lwtformer_modules import LearnableHaarDWT2D, _groups


class ChannelPreservingWaveDownLite(nn.Module):
    """DW downsampling modulated by a learnable, per-channel wavelet gate.

    No dense C->C/4 convolution at full resolution or dense 4C->C fusion.
    The four subbands of EACH input channel are fused independently. Channel
    interaction remains in the following HG_Block. Output spatial size matches
    the original stride-2, padding-1 backbone convolution, including odd inputs.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}.')
        # Keep normalization names explicit for the existing optimizer regexes.
        self.main = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(channels, channels, 3, stride=2, padding=1,
                               groups=channels, bias=False)),
            ('norm', nn.BatchNorm2d(channels)),
        ]))
        self.dwt = LearnableHaarDWT2D()
        self.band_mix = nn.Conv2d(4 * channels, channels, 1,
                                  groups=channels, bias=True)
        self.gate = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(channels, channels, 3, padding=1,
                               groups=channels, bias=False)),
            ('norm', nn.GroupNorm(_groups(channels), channels)),
            ('act', nn.Sigmoid()),
        ]))

    def forward(self, x):
        main = self.main(x)
        # DWT uses an unpadded stride-2 filter; pad its input to match ceil(H/2).
        pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
        wave_input = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate') \
            if pad_h or pad_w else x
        ll, lh, hl, hh = self.dwt(wave_input)
        # Interleave [LL_c, LH_c, HL_c, HH_c] for grouped 4->1 fusion.
        # torch.cat along C would incorrectly mix different input channels.
        bands = torch.stack((ll, lh, hl, hh), dim=2).flatten(1, 2)
        gate = self.gate(self.band_mix(bands))
        return main * (1.0 + gate)
