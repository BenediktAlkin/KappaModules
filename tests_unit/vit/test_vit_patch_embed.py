import unittest

import einops
import torch
from timm.layers.patch_embed import PatchEmbed as TimmPatchEmbed

from kappamodules.vit import VitPatchEmbed


class TestVitPatchEmbed(unittest.TestCase):
    def test_eq_timm(self):
        torch.manual_seed(0)
        embed_km = VitPatchEmbed(dim=7, num_channels=2, resolution=(32, 32), patch_size=4, init_weights="torch")
        torch.manual_seed(0)
        embed_og = TimmPatchEmbed(img_size=32, patch_size=4, in_chans=2, embed_dim=7)
        x = torch.randn(1, 2, 32, 32)
        y_km = embed_km(x)
        y_og = embed_og(x)
        self.assertEqual((1, 8, 8, 7), y_km.shape)
        self.assertTrue(torch.all(einops.rearrange(y_km, "b ... d -> b (...) d") == y_og))
