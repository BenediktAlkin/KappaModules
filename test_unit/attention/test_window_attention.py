import unittest

import torch

from kappamodules.attention.window_attention import WindowAttention
from original_modules.original_window_attention import OriginalWindowAttention

class TestWindowAttention(unittest.TestCase):
    def test_equal_to_original(self):
        kwargs = dict(dim=4, window_size=(3, 3), num_heads=2, qkv_bias=True)
        torch.manual_seed(98243)
        new = WindowAttention(**kwargs)
        torch.manual_seed(98243)
        og = OriginalWindowAttention(**kwargs)

        x = torch.randn(5, 9, 4, generator=torch.Generator().manual_seed(8493))
        new_x = new(x)
        og_x = og(x)

        for (og_name, og_param), (new_name, new_param) in zip(new.named_parameters(), og.named_parameters()):
            self.assertEqual(og_name, new_name)
            self.assertTrue(torch.all(og_param == new_param))

        self.assertTrue(torch.allclose(og_x, new_x))

    def test_1d(self):
        torch.manual_seed(98243)
        new = WindowAttention(dim=4, window_size=(3,), num_heads=2, qkv_bias=True)

        x = torch.randn(5, 3, 4, generator=torch.Generator().manual_seed(8493))
        new_x = new(x)
        # TODO

    def test_3d(self):
        torch.manual_seed(98243)
        new = WindowAttention(dim=4, window_size=(3, 3, 3), num_heads=2, qkv_bias=True)

        x = torch.randn(5, 27, 4, generator=torch.Generator().manual_seed(8493))
        new_x = new(x)
        # TODO


