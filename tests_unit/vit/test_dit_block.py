import unittest

import torch

from kappamodules.vit import DitBlock
from original_modules.original_dit_block import OriginalDitBlock

class TestDitBlock(unittest.TestCase):
    def test(self):
        dim = 12
        seed = 239084
        torch.manual_seed(seed)
        block = DitBlock(dim=dim, num_heads=2, init_weights="torch")
        torch.manual_seed(seed)
        block_og = OriginalDitBlock(hidden_size=dim, num_heads=2)

        x = torch.randn(2, 6, dim, generator=torch.Generator().manual_seed(9834))
        cond = torch.randn(2, dim, generator=torch.Generator().manual_seed(564))
        y = block(x, cond)
        y_og = block_og(x, cond)
        self.assertEqual(x.shape, y.shape)