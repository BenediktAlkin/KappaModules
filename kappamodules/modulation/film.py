from torch import nn
import math
from kappamodules.utils.shapes import to_ndim

class Film(nn.Module):
    def __init__(self, cond_dim, out_dim, init="xavier_uniform"):
        super().__init__()
        self.in_dim = cond_dim
        self.out_dim = out_dim
        self.modulation = nn.Linear(cond_dim, out_dim)
        self.init = init
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.modulation.weight.shape[1]
        assert self.modulation.weight.shape[0] % 2 == 0
        fan_out = self.modulation.weight.shape[0] // 2
        if self.init == "xavier_uniform":
            val = math.sqrt(6 / (fan_out + fan_in))
            nn.init.uniform_(self.modulation.weight, -val, val)
            nn.init.zeros_(self.modulation.bias)
        else:
            raise NotImplementedError

    def forward(self, x, cond):
        scale, shift = self.modulation(to_ndim(cond, ndim=x.ndim)).chunk(2, dim=1)
        return x * (scale + 1) + shift