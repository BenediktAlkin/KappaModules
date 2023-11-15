from torch import nn

from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias, init_norms_as_noaffine
from kappamodules.layers import WeightNormLinear, Normalize
from kappamodules.utils.mode_to_ctor import mode_to_norm_ctor


class DinoProjector(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim=2048,
            num_hidden_layers=1,
            bottleneck_dim=256,
            output_dim=65536,
            fixed_g=True,
            norm_mode="none",
            init_weights="xavier_uniform",
            eps=1e-6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.norm_mode = norm_mode
        self.fixed_g = fixed_g
        self.init_weights = init_weights
        self.eps = eps

        # ctors
        self.norm_ctor, self.requires_bias = mode_to_norm_ctor(norm_mode)
        self.act_ctor = nn.GELU

        # initial layer
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=self.requires_bias),
            self.norm_ctor(hidden_dim, eps=eps),
            self.act_ctor(),
        )
        # hidden layers
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=self.requires_bias),
                    self.norm_ctor(hidden_dim, eps=eps),
                    self.act_ctor(),
                )
                for _ in range(num_hidden_layers)
            ]
        )
        # bottleneck
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        # last layer
        self.last_layer = nn.Sequential(
            Normalize(),
            WeightNormLinear(bottleneck_dim, output_dim, fixed_g=fixed_g, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        self.apply(init_norms_as_noaffine)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.bottleneck(x)
        x = self.last_layer(x)
        return x
