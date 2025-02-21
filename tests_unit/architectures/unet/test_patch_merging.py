import unittest

from kappamodules.architectures.unet.patch_merging import PatchMergingNd
import torch

class TestPatchMerging(unittest.TestCase):
    def _test_shape(self, seqlens, channel_first):
        input_dim = 5
        output_dim = 7
        half_seqlens = [seqlen // 2 for seqlen in seqlens]
        merge = PatchMergingNd(
            input_dim=input_dim,
            output_dim=output_dim,
            ndim=len(seqlens),
            channel_first=channel_first,
        )
        x = torch.randn(1, input_dim, *seqlens)
        y = merge(x)
        if channel_first:
            expected_shape = (1, output_dim, *half_seqlens)
        else:
            expected_shape = (1, *half_seqlens, output_dim)
        self.assertEqual(y.shape, expected_shape)

    def test_shape_2d_cfirst(self):
        self._test_shape(
            seqlens=(6, 4),
            channel_first=True,
        )

    def test_shape_2d_clast(self):
        self._test_shape(
            seqlens=(6, 4),
            channel_first=False,
        )

    def test_shape_3d_cfirst(self):
        self._test_shape(
            seqlens=(6, 4, 8),
            channel_first=True,
        )

    def test_shape_4d_cfirst(self):
        self._test_shape(
            seqlens=(6, 4, 8, 2),
            channel_first=True,
        )
