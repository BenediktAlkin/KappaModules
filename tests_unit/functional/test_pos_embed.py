import unittest
from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens

class TestPosEmbed(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual((10, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10,), dim=5).shape)
        self.assertEqual((10, 20, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10, 20), dim=5).shape)
        self.assertEqual((10, 20, 30, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10, 20, 30), dim=5).shape)
