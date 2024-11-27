import unittest

import torch

from kappamodules.attention import PerceiverAttentionLocalgrid2d
from torch import nn

class TestPerceiverAttentionLocalgrid(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(9823)
        dim = 24
        seqlen_h = 6
        seqlen_w = 7
        attn = PerceiverAttentionLocalgrid2d(
            kernel_size=3,
            dim=dim,
            num_heads=4,
        )
        x = torch.randn(1, seqlen_h, seqlen_w, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)


    def test_numbers(self):
        torch.manual_seed(9823)
        seqlen_h = 3
        seqlen_w = 4
        attn = PerceiverAttentionLocalgrid2d(
            kernel_size=3,
            dim=1,
            num_heads=1,
        )
        x = torch.stack(
            torch.meshgrid(
                torch.arange(seqlen_h).float(),
                torch.arange(seqlen_w).float(),
                indexing="ij",
            )
        )
        x = (x[1] + x[0] * seqlen_w).unsqueeze(0).unsqueeze(-1) + 1
        nn.init.ones_(attn.qkv.weight)
        nn.init.zeros_(attn.qkv.bias)
        nn.init.ones_(attn.proj.weight)
        nn.init.zeros_(attn.proj.bias)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(torch.tensor(118.42804718017578), y.sum()), y.sum().item())
