import torch
import unittest
from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens

class TestPosEmbed(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual((10, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10,), dim=5).shape)
        self.assertEqual((10, 20, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10, 20), dim=5).shape)
        self.assertEqual((10, 20, 30, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10, 20, 30), dim=5).shape)

    def test_padding_sincospad_2d(self):
        pos_embed_padded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=15)
        # pos_embed_unpadded is still padded because dim needs to be divided per ndim and then also between sin/cos
        pos_embed_unpadded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=14)
        self.assertEqual((2, 3, 15), pos_embed_padded.shape)
        self.assertTrue(torch.all(pos_embed_padded[:, :, -1] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :-1] == pos_embed_unpadded))

    def test_nodimpadding_nosincospad_2d(self):
        pos_embed_padded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=9)
        pos_embed_unpadded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=8)
        self.assertEqual((2, 3, 9), pos_embed_padded.shape)
        self.assertTrue(torch.all(pos_embed_padded[:, :, -1] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :-1] == pos_embed_unpadded))

    def test_nodimpadding_nosincospad_3d(self):
        pos_embed_padded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3, 4), dim=26)
        pos_embed_unpadded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3, 4), dim=24)
        self.assertEqual((2, 3, 4, 26), pos_embed_padded.shape)
        self.assertTrue(torch.all(pos_embed_padded[:, :, :, -1] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :, -2] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :, :-2] == pos_embed_unpadded))