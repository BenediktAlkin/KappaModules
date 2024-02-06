from torch import nn

from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from kappamodules.utils.param_checking import to_2tuple


class Mlp(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim=None,
            out_dim=None,
            act_ctor=nn.GELU,
            bias=True,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
    ):
        super().__init__()
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        bias1, bias2 = to_2tuple(bias)

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias1)
        self.act = act_ctor()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias2)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
