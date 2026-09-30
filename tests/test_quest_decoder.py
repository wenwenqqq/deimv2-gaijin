"""Run: python -m unittest discover -s tests -p test_quest_decoder.py"""

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.core import YAMLConfig
from engine.core.workspace import create
from engine.deim.deim_decoder import DEIMTransformer
from engine.deim.deim_decoder_quest import DEIMTransformer_QUEST, QUESTQueryAttention


class QUESTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_formula_and_masks(self):
        module = QUESTQueryAttention(16, 4, batch_first=True).double().eval()
        x = torch.randn(2, 5, 16, dtype=torch.float64)
        blocked = torch.zeros(5, 5, dtype=torch.bool)
        blocked[:2, 2:] = True
        blocked[2:, :2] = True
        additive = torch.zeros(5, 5, dtype=torch.float64).masked_fill(blocked, -torch.inf)
        q, k, v = F.linear(x, module.in_proj_weight, module.in_proj_bias).chunk(3, -1)
        q, k, v = [t.reshape(2, 5, 4, 4).transpose(1, 2) for t in (q, k, v)]
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        expected = ((q @ k.transpose(-1, -2) + additive).softmax(-1) @ v)
        expected = module.out_proj(expected.transpose(1, 2).reshape(2, 5, 16))
        for mask in (blocked, additive, blocked.expand(8, -1, -1)):
            actual, _ = module(x, x, x, attn_mask=mask)
            torch.testing.assert_close(actual, expected)
        # Perturb forbidden DN tokens: matching query outputs must not change.
        changed = x.clone()
        changed[:, :2] += 20
        out, _ = module(changed, changed, changed, attn_mask=blocked)
        torch.testing.assert_close(out[:, 2:], expected[:, 2:])

    def test_key_norm_invariance_and_zero_keys(self):
        module = QUESTQueryAttention(16, 4, batch_first=True).eval()
        with torch.no_grad():
            module.in_proj_bias.zero_()
        q, k, v = [torch.randn(2, 5, 16) for _ in range(3)]
        original, _ = module(q, k, v)
        scaled, _ = module(q, k * torch.rand(2, 5, 1).add(0.5), v)
        torch.testing.assert_close(original, scaled)
        out, _ = module(q, torch.zeros_like(k), v)
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(module.in_proj_weight.grad).all())

    def test_initialization_and_cross_attention_unchanged(self):
        args = dict(hidden_dim=32, feat_channels=[32, 32], feat_strides=[16, 32],
                    num_levels=2, num_points=[2, 2], num_layers=3, dim_feedforward=64)
        torch.manual_seed(7)
        baseline = DEIMTransformer(**args)
        rng_after_baseline = torch.get_rng_state()
        torch.manual_seed(7)
        quest = DEIMTransformer_QUEST(**args)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_baseline))
        self.assertEqual(baseline.state_dict().keys(), quest.state_dict().keys())
        for name, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, quest.state_dict()[name], rtol=0, atol=0)
        for before, after in zip(baseline.decoder.layers, quest.decoder.layers):
            self.assertIsInstance(after.self_attn, QUESTQueryAttention)
            self.assertIs(type(before.cross_attn), type(after.cross_attn))

    def test_registered_config_training_with_denoising(self):
        path = Path(__file__).resolve().parents[1] / 'configs/deimv2/deimv2_hgnetv2_moe_v2_n_uavdt_HSFPN_dw_QUEST_decoder.yml'
        cfg = YAMLConfig(str(path))
        model = create('DEIMTransformer_QUEST', cfg.global_cfg).train()
        self.assertEqual(len(model.decoder.layers), 3)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(cfg.yaml_cfg['moe_loss_weight'], 0.0)
        features = [torch.randn(2, 128, 20, 20), torch.randn(2, 128, 10, 10)]
        targets = [dict(labels=torch.tensor([0, 1]),
                        boxes=torch.tensor([[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]))
                   for _ in range(2)]
        out = model(features, targets)
        self.assertIn('dn_outputs', list(out))
        self.assertEqual(out['pred_logits'].shape, (2, 300, 3))
        loss = out['pred_logits'].square().mean() + out['pred_boxes'].square().mean()
        loss.backward()
        for layer in model.decoder.layers:
            grad = layer.self_attn.in_proj_weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_zero_keys_backward(self):
        module = QUESTQueryAttention(128, 8, batch_first=True).cuda().train()
        with torch.no_grad():
            module.in_proj_bias.zero_()
        q = torch.randn(2, 32, 128, device='cuda', requires_grad=True)
        mask = torch.zeros(32, 32, dtype=torch.bool, device='cuda')
        mask[16:, :16] = True
        with torch.autocast('cuda', dtype=torch.float16):
            out, _ = module(q, torch.zeros_like(q), q, attn_mask=mask)
            loss = out.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(module.in_proj_weight.grad).all())


if __name__ == '__main__':
    unittest.main()
