import math

from torch import nn


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
        fan_in = self.modulation.weight.shape[1]
        assert self.modulation.weight.shape[0] % self.num_outputs == 0
        fan_out = self.modulation.weight.shape[0] // self.num_outputs
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            val = math.sqrt(6 / (fan_out + fan_in))
            nn.init.uniform_(self.modulation.weight, -val, val)
            nn.init.zeros_(self.modulation.bias)
        else:
            raise NotImplementedError

    def forward(self, cond):
        # exmple: transformer block
        # scale1, shift1, gate1, scale2, shift2, gate2 = self.modulation(cond).chunk(6, dim=1)
        return self.modulation(cond).chunk(self.num_outputs, dim=1)
