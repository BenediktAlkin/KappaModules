import math

from torch import nn

from kappamodules.utils.shapes import to_ndim

from kappamodules.init import init_xavier_uniform_merged_linear

class Film(nn.Module):
    def __init__(self, dim_cond, dim_out, init_weights="xavier_uniform"):
        super().__init__()
        self.dim_cond = dim_cond
        self.dim_out = dim_out
        self.modulation = nn.Linear(dim_cond, dim_out * 2)
        self.init_weights = init_weights
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            init_xavier_uniform_merged_linear(self.modulation, num_layers=2)
            nn.init.zeros_(self.modulation.bias)
        else:
            raise NotImplementedError

    def forward(self, x, cond):
        scale, shift = self.modulation(to_ndim(cond, ndim=x.ndim)).chunk(2, dim=1)
        return x * (scale + 1) + shift
