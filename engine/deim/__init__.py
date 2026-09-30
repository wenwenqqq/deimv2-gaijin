"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .deim import DEIM

from .matcher import HungarianMatcher

from .hybrid_encoder import HybridEncoder
from .lite_encoder import LiteEncoder
from .hybrid_encoder_sfa import HybridEncoder_SFA
from .hybrid_encoder_sfa_G1gate import HybridEncoder_SFA_G1gate
# from .hybrid_encoder_SLA import HybridEncoder_SLA
from .hybrid_encoder_LCGA import HybridEncoder_LCGA
from .hybrid_encoder_TransMixer import HybridEncoder_TransMixer
from .hybrid_encoder_PFGA import HybridEncoder_PFGA
from .hybrid_encoder_HSFPN import HybridEncoder_HSFPN
from .hybrid_encoder_lwthfp import HybridEncoder_LWTHFP
from .hybrid_encoder_Combo import HybridEncoder_Combo
from .hybrid_encoder_drfd import HybridEncoderDRFD



from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion
from .deim_decoder import DEIMTransformer
from .deim_decoder_lwtsea import DEIMTransformer_LWTSEA
from .deim_decoder_quest import DEIMTransformer_QUEST
from .deim_decoder_quest_sea import DEIMTransformer_QUEST_SEA
from .deim_decoder_lwga import DEIMTransformer_LWGA
from .deim_decoder_nsa import DEIMTransformer_NSA
