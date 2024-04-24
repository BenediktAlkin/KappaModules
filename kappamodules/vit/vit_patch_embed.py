import einops
import numpy as np
from torch import nn

from kappamodules.init import init_truncnormal_zero_bias
from kappamodules.utils.param_checking import to_ntuple


class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, init_weights="xavier_uniform"):
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        for i in range(self.ndim):
            assert resolution[i] % self.patch_size[i] == 0, f"resolution[{i}] % patch_size[{i}] != 0"
        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        # use primitive type as np.prod gives np.int which can is not compatible with all serialization/logging
        self.num_patches = int(np.prod(self.seqlens))

        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError

        self.proj = conv_ctor(num_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "xavier_uniform":
            # initialize as nn.Linear
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim))
        x = self.proj(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x
