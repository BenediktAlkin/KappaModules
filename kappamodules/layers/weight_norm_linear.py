import math

import torch
import torch.nn.functional as F
from torch import nn


class WeightNormLinear(nn.Module):
    """
    torch.nn.utils.weight_norm(nn.Linear(...)) but with weight_g as buffer when it is fixed
    if weight_g is set to requires_grad=False (as done in DINO, iBOT, MUGS, ...) and during
    training a parent module is unfrozen via
    ```
    for p in module.parameters():
      p.requires_grad = True
    ```
    the weight_g is also unfrozen. registering weight_g as a buffer avoids this
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            fixed_g: bool = False,
            device=None,
            dtype=None,
            init_weights="xavier_uniform",
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_g = fixed_g
        self.init_weights = init_weights
        self.weight_v = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if fixed_g:
            self.register_buffer("weight_g", torch.empty(out_features, 1, **factory_kwargs))
        else:
            self.weight_g = nn.Parameter(torch.empty(out_features, 1, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # normal initialization
        if self.init_weights == "torch":
            nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(5))
            if self.bias is not None:
                # noinspection PyProtectedMember
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_v)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        elif self.init_weights == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight_v)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.)

        # weight norm intialization
        with torch.no_grad():
            if self.fixed_g:
                self.weight_g.fill_(1)
            else:
                self.weight_g.copy_(torch.norm_except_dim(self.weight_v))

    def extra_repr(self):
        return nn.Linear.extra_repr(self)

    def forward(self, x):
        # noinspection PyProtectedMember
        return F.linear(x, torch._weight_norm(self.weight_v, self.weight_g, 0), self.bias)
