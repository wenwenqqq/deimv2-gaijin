"""Run with: python -m unittest discover -s tests -p test_wavedown_lite.py"""

import unittest

import torch

from engine.deim.lwtformer_modules_lite import ChannelPreservingWaveDownLite


class WaveDownLiteTests(unittest.TestCase):
    def test_even_and_odd_shapes(self):
        module = ChannelPreservingWaveDownLite(8).eval()
        for height, width in [(16, 16), (15, 17), (16, 17)]:
            with self.subTest(height=height, width=width):
                y = module(torch.randn(2, 8, height, width))
                self.assertEqual(tuple(y.shape), (2, 8, (height + 1) // 2, (width + 1) // 2))
                self.assertTrue(torch.isfinite(y).all().item())

    def test_subbands_are_fused_per_input_channel(self):
        module = ChannelPreservingWaveDownLite(8).eval()
        # Select only LL in each channel; a band-major concatenation fails this.
        with torch.no_grad():
            module.band_mix.weight.zero_()
            module.band_mix.weight[:, 0, 0, 0] = 1.
            module.band_mix.bias.zero_()
        x = torch.randn(2, 8, 16, 16)
        seen = []
        hook = module.band_mix.register_forward_hook(lambda mod, args, out: seen.append(out))
        try:
            module(x)
        finally:
            hook.remove()
        torch.testing.assert_close(seen[0], module.dwt(x)[0])

    def test_wavelet_and_main_gradients(self):
        torch.manual_seed(0)
        module = ChannelPreservingWaveDownLite(8).train()
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        module(x).square().mean().backward()
        for name, param in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all().item())
        for param in [module.main[0].weight, module.dwt.dec_lo,
                      module.dwt.dec_hi, module.band_mix.weight]:
            self.assertGreater(param.grad.abs().sum().item(), 0.)

    def test_stage3_parameter_budget(self):
        module = ChannelPreservingWaveDownLite(256)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 6916)

    def test_cpu_autocast_backward(self):
        module = ChannelPreservingWaveDownLite(8).train()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            out = module(torch.randn(2, 8, 16, 16))
            loss = out.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(module.dwt.dec_lo.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_amp_backward(self):
        module = ChannelPreservingWaveDownLite(8).cuda().train()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = module(torch.randn(2, 8, 16, 16, device='cuda'))
            loss = out.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(module.dwt.dec_hi.grad).all().item())


if __name__ == '__main__':
    unittest.main()
