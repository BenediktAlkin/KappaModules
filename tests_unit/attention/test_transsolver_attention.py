import unittest

import einops
import numpy as np
import torch

from kappamodules.attention import TranssolverAttention


class TestDotProductAttention(unittest.TestCase):
    def test_unmasked(self):
        torch.manual_seed(98234)
        dim = 16
        attn = TranssolverAttention(
            dim=dim,
            num_heads=2,
            num_slices=3,
            qkv_bias=False,
            init_weights="xavier_uniform",
        )
        seqlen = 13
        x = torch.randn(2, seqlen, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        # unmerged qkv
        # self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.002519140485674143)))
        # merged qkv
        # self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.0014637971762567759)))
        # merged qkv + split xavier init
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.0017974275397136807)))

    def test_masked(self):
        torch.manual_seed(98234)
        dim = 16
        attn = TranssolverAttention(
            dim=dim,
            num_heads=2,
            num_slices=3,
            qkv_bias=False,
            init_weights="xavier_uniform",
        )
        batch_size = 2
        seqlen = 13
        x = torch.randn(2, seqlen, dim)
        mask = torch.ones(batch_size, seqlen, dtype=torch.bool)
        mask[0, 10:] = False
        y = attn(x, attn_mask=mask)
        self.assertEqual(x.shape, y.shape)
        # merged qkv + split xavier init
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.0024041568394750357)))
        # rescale masked outputs to make sure they are not included in attention
        x[0, 10:] = 1000000000
        y = attn(x, attn_mask=mask)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.0024041568394750357)))
