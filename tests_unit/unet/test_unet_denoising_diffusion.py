import torch
import unittest

from kappamodules.unet import UnetDenoisingDiffusion

class TestUnetDenoisingDiffusion(unittest.TestCase):
    def test_1d_uncond(self):
        torch.manual_seed(9823)
        num_channels = 1
        seqlen = 8
        model = UnetDenoisingDiffusion(dim=8, ndim=1, dim_in=num_channels, dim_out=num_channels, num_heads=2, depth=2)
        x = torch.randn(2, num_channels, seqlen)
        y = model(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.22751933336257935)))