import unittest

from kappamodules.attention import PerceiverAttention1d
from original_modules.original_perceiver_attention import OriginalPerceiverAttention
import torch

class TestPerceiverAttention(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(9823)
        dim = 24
        seqlen_q = 6
        seqlen_kv = 7
        attn = PerceiverAttention1d(dim=dim, num_heads=4)
        q = torch.randn(1, seqlen_q, dim)
        kv = torch.randn(1, seqlen_kv, dim)
        y = attn(q=q, kv=kv)
        self.assertEqual(q.shape, y.shape)

    def test_equal_to_original(self):
        dim = 4
        seqlen_q = 2
        seqlen_kv = 3
        num_heads = 1
        torch.manual_seed(9823)
        attn_kc = PerceiverAttention1d(dim=dim, num_heads=num_heads, init_weights="torch")
        torch.manual_seed(9823)
        attn_og = OriginalPerceiverAttention(dim=dim, dim_head=dim // num_heads, heads=num_heads)
        torch.manual_seed(9823)
        q = torch.randn(1, seqlen_q, dim)
        kv = torch.randn(1, seqlen_kv, dim)
        y_kc = attn_kc(q=q, kv=kv)
        y_og = attn_og(latents=q, x=kv)
        self.assertEqual(q.shape, y_og.shape)
        self.assertEqual(y_kc.shape, y_og.shape)
        self.assertTrue(torch.allclose(y_kc, y_og))

