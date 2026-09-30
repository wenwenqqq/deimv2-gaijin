import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from engine.deim.hybrid_encoder_TransMixer import HybridEncoder_TransMixer

# 创建编码器
encoder = HybridEncoder_TransMixer(
    use_encoder_idx=[1, 2],  # 小目标配置
    mlp_ratio=2.0,
    state_dim=32
)

# 模拟输入特征（来自backbone的输出）
feats = [
    torch.randn(2, 512, 80, 80),   # 8x下采样
    torch.randn(2, 1024, 40, 40),  # 16x下采样
    torch.randn(2, 2048, 20, 20)   # 32x下采样
]

# 前向传播
outputs = encoder(feats)

# 检查输出形状
for i, out in enumerate(outputs):
    print(f"输出特征{i}形状: {out.shape}")
    assert out.shape == (2, 256, 80//(2**i), 80//(2**i)), f"输出形状错误"

print("✅ TransMixer替换成功，前向传播正常！")