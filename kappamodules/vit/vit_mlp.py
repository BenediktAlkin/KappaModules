from torch import nn

from kappamodules.init.functional import initialize_xavier_uniform_zero_bias
from kappamodules.utils.param_checking import to_2tuple


class VitMlp(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim=None,
            out_dim=None,
            act_ctor=nn.GELU,
            bias=True,
            init="xavier_uniform",
    ):
        super().__init__()
        self.init = init
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        bias1, bias2 = to_2tuple(bias)

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias1)
        self.act = act_ctor()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias2)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init == "torch":
            pass
        elif self.init == "xavier_uniform":
            self.apply(initialize_xavier_uniform_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
