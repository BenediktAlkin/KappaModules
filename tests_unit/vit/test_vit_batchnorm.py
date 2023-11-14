import unittest

from kappamodules.vit import VitBatchNorm
import torch

class TestVitBatchNorm(unittest.TestCase):
    def test_0d(self):
        dim = 4
        bn = VitBatchNorm(num_features=dim, eps=0)
        x = torch.rand(16, dim, generator=torch.Generator().manual_seed(89543))
        y = bn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.allclose(y.mean(dim=0), torch.zeros(size=(dim,)), atol=1e-6))
        self.assertTrue(torch.allclose(y.std(dim=0), torch.ones(size=(dim,)), atol=1e-1))

    def test_1d(self):
        dim = 4
        bn = VitBatchNorm(num_features=dim, eps=0)
        x = torch.rand(16, 4, dim, generator=torch.Generator().manual_seed(89543))
        y = bn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.allclose(y.mean(dim=0), torch.zeros(size=(dim,)), atol=1e-6))
        self.assertTrue(torch.allclose(y.std(dim=0), torch.ones(size=(dim,)), atol=1e-1))