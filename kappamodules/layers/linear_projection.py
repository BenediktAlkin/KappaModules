from torch import nn

from kappamodules.init import init_xavier_uniform_zero_bias


class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, init_weights="xavier_uniform"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.init_weights = init_weights
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "nonaffine":
            init_xavier_uniform_zero_bias(self.proj)
        else:
            raise NotImplementedError
