import unittest

import einops
import torch

from kappamodules.vit import VitClassTokens


class TestVitClassToken(unittest.TestCase):
    def test_first(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=1, location="first")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        self.assertTrue(torch.all(einops.repeat(cls_token.tokens, "1 1 dim -> bs dim", bs=len(x)) == pooled))
        self.assertTrue(torch.all(y[:, 0] == pooled))

    def test_middle(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=1, location="middle")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        self.assertTrue(torch.all(einops.repeat(cls_token.tokens, "1 1 dim -> bs dim", bs=len(x)) == pooled))
        self.assertTrue(torch.all(y[:, 2] == pooled))

    def test_middle_vit(self):
        torch.manual_seed(0)
        x = torch.randn(2, 196, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=1, location="middle")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        self.assertTrue(torch.all(einops.repeat(cls_token.tokens, "1 1 dim -> bs dim", bs=len(x)) == pooled))
        self.assertTrue(torch.all(y[:, 98] == pooled))
