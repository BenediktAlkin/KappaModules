import unittest
import torch
from kappamodules.layers import ContinuousSincosEmbed

class TestContinuousSincosEmbed(unittest.TestCase):
    def test_shape(self):
        batch_size = 3
        num_points = 32
        ndim = 2
        dim = 16
        coords = torch.rand(batch_size, num_points, ndim, generator=torch.Generator().manual_seed(8943))
        embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        result = embed(coords)
        self.assertEqual((batch_size, num_points, dim), result.shape)