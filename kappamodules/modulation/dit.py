import math

from torch import nn
from kappamodules.init.functional import init_truncnormal_zero_bias, init_xavier_uniform_merged_linear


class Dit(nn.Module):
    def __init__(self, cond_dim, out_dim, num_outputs=6, init_weights="xavier_uniform"):
        super().__init__()
        self.in_dim = cond_dim
        self.out_dim = out_dim
        self.num_outputs = num_outputs
        self.modulation = nn.Linear(cond_dim, num_outputs * out_dim)
        self.init_weights = init_weights
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            init_xavier_uniform_merged_linear(self.modulation, num_layers=self.num_outputs)
        elif self.init_weights == "truncnormal":
            init_truncnormal_zero_bias(self.modulation)
        else:
            raise NotImplementedError

    def forward(self, cond):
        # exmple: transformer block
        # scale1, shift1, gate1, scale2, shift2, gate2 = self.modulation(cond).chunk(6, dim=1)
        return self.modulation(cond).chunk(self.num_outputs, dim=1)
