"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .common import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .presnet import PResNet
from .test_resnet import MResNet

from .timm_model import TimmModel
from .torchvision_model import TorchVisionModel

from .csp_resnet import CSPResNet
from .csp_darknet import CSPDarkNet, CSPPAN

from .hgnetv2 import HGNetv2
from .hgnetv2_delta import HGNetv2_delta
from .hgnetv2_sfa import HGNetv2_SFA

# ES-MoE Enhanced HGNetv2 (from YOLO-Master)
from .hgnetv2_moe import HGNetv2_MoE
from .hgnetv2_moe_v2 import HGNetv2_MoE_v2
from .hgnetv2_moe_v2_wavedown import HGNetv2_MoE_v2_WaveDown
from .hgnetv2_moe_v2_wavedown_lite import HGNetv2_MoE_v2_WaveDownLite

from .dinov3_adapter import *
