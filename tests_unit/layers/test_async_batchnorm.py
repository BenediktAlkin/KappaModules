import unittest

import torch
from torch import nn

from kappamodules.layers import AsyncBatchNorm


class TestAsyncBatchNorm(unittest.TestCase):
    def test_abn_shape(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        asb = AsyncBatchNorm(dim=10)
        y = asb(x)
        self.assertEqual(x.shape, y.shape)

    def test_abn_converges(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        asb = AsyncBatchNorm(dim=10, momentum=0.0)
        bn = torch.nn.BatchNorm1d(10)
        # momentum=0 -> track stats
        _ = asb(x)
        # normalize with stats
        y2 = asb(x)
        # compate to batchnorm
        y3 = bn(x)
        self.assertTrue(torch.allclose(y2.std(), y3.std()))

    def test_abn_backward(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        asb = AsyncBatchNorm(dim=10)
        y = asb(x)
        y.mean().backward()
        self.assertEqual(x.shape, y.shape)

    def test_convert(self):
        syncbn = nn.Sequential(nn.Linear(5, 10), nn.BatchNorm1d(10))
        asyncbn = AsyncBatchNorm.convert_async_batchnorm(syncbn)
        self.assertIsInstance(asyncbn[1], AsyncBatchNorm)
