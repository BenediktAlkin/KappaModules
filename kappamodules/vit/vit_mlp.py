from torch import nn


class VitMlp(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim=None,
            out_dim=None,
            act_ctor=nn.GELU,
            bias=True,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.act = act_ctor()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
