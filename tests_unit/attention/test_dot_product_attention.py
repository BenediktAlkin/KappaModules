import unittest

import torch

from kappamodules.attention import DotProductAttention1d, DotProductAttention2d, DotProductAttention3d


class TestDotProductAttention(unittest.TestCase):
    def test_1d_channellast(self):
        torch.manual_seed(98234)
        dim = 16
        attn = DotProductAttention1d(dim=dim, num_heads=2, qkv_bias=True, init_weights="xavier_uniform")
        seqlen = 13
        x = torch.randn(2, seqlen, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.010636774823069572)))

    def test_2d_channellast(self):
        torch.manual_seed(98234)
        dim = 16
        attn = DotProductAttention2d(dim=dim, num_heads=2, qkv_bias=True, init_weights="xavier_uniform")
        x = torch.randn(2, dim, 2, 3)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.047952961176633835)))

    def test_3d_channellast(self):
        torch.manual_seed(98234)
        dim = 16
        attn = DotProductAttention3d(dim=dim, num_heads=2, qkv_bias=True, init_weights="xavier_uniform")
        x = torch.randn(2, dim, 2, 3, 4)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.019655022770166397)))
