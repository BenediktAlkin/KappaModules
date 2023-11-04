import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from kappamodules.drop_path import DropPath
from original_modules.original_drop_path import OriginalDropPath

class TestEfficientDropPath(unittest.TestCase):
    def is_equal_to_original(self, dim, drop_prob, stochastic_size, scale_by_keep, training, seed):
        # create modules
        torch.manual_seed(seed)
        custom_layer = nn.Linear(dim, dim)
        torch.manual_seed(seed)
        original_layer = nn.Linear(dim, dim)
        kwargs = dict(drop_prob=drop_prob, stochastic_size=stochastic_size, scale_by_keep=scale_by_keep)
        custom = DropPath(custom_layer, **kwargs)
        original = OriginalDropPath(**kwargs)
        # prepare forward
        if not training:
            custom = custom.eval()
            original = original.eval()

        # create data (and make sure that DropPath does nothing inplace)
        x = torch.randn(10, dim, generator=torch.Generator().manual_seed(seed))
        og_x = x.clone()
        # original forward + backward
        torch.manual_seed(seed)
        original_y = x + original(original_layer(x))
        original_y.mean().backward()
        # custom forward
        torch.manual_seed(seed)
        custom_y = custom(x)
        custom_y.mean().backward()
        # check
        self.assertTrue(torch.all(x == og_x))
        self.assertTrue(torch.allclose(custom_y, original_y))
        self.assertTrue(torch.allclose(original_layer.weight.grad, custom_layer.weight.grad))
        self.assertTrue(torch.allclose(original_layer.bias.grad, custom_layer.bias.grad))

    def test_03_stochasticsize_scalebykeep_training(self):
        self.is_equal_to_original(
            dim=4,
            drop_prob=0.3,
            stochastic_size=True,
            scale_by_keep=True,
            training=True,
            seed=4389,
        )

    def test_03_fixedsize_scalebykeep_training(self):
        self.is_equal_to_original(
            dim=4,
            drop_prob=0.3,
            stochastic_size=False,
            scale_by_keep=True,
            training=True,
            seed=4389,
        )

    def test_03_stochasticsize_noscalebykeep_training(self):
        self.is_equal_to_original(
            dim=4,
            drop_prob=0.3,
            stochastic_size=True,
            scale_by_keep=False,
            training=True,
            seed=4389,
        )

    def test_03_stochasticsize_scalebykeep_eval(self):
        self.is_equal_to_original(
            dim=4,
            drop_prob=0.3,
            stochastic_size=True,
            scale_by_keep=True,
            training=False,
            seed=4389,
        )

    def test_03_fixedsize_noscalebykeep_training(self):
        self.is_equal_to_original(
            dim=4,
            drop_prob=0.3,
            stochastic_size=False,
            scale_by_keep=False,
            training=True,
            seed=4389,
        )

    def test_03_fixedsize_scalebykeep_eval(self):
        self.is_equal_to_original(
            dim=4,
            drop_prob=0.3,
            stochastic_size=False,
            scale_by_keep=True,
            training=False,
            seed=4389,
        )

    def test_03_fixedsize_noscalebykeep_eval(self):
        self.is_equal_to_original(
            dim=4,
            drop_prob=0.3,
            stochastic_size=False,
            scale_by_keep=False,
            training=False,
            seed=4389,
        )
