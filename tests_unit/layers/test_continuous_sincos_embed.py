import unittest
import torch
from kappamodules.layers import ContinuousSincosEmbed

class TestContinuousSincosEmbed(unittest.TestCase):
    def test_shape(self):
        batch_size = 3
        num_points = 32
        ndim = 2
        dim = 16
        # dense
        dense_coords = torch.rand(batch_size, num_points, ndim, generator=torch.Generator().manual_seed(8943))
        embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        dense_result = embed(dense_coords)
        self.assertEqual((batch_size, num_points, dim), dense_result.shape)
        # sparse
        sparse_coords = torch.rand(batch_size * num_points, ndim, generator=torch.Generator().manual_seed(8943))
        embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        sparse_result = embed(sparse_coords)
        self.assertEqual((batch_size * num_points, dim), sparse_result.shape)