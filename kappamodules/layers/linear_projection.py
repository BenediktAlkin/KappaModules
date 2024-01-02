from torch import nn

from kappamodules.init import init_xavier_uniform_zero_bias


class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, ndim=None, bias=True, init_weights="xavier_uniform"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.bias = bias
        self.init_weights = init_weights
        if ndim is None:
            self.proj = nn.Linear(input_dim, output_dim, bias=bias)
        elif ndim == 1:
            self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=bias)
        elif ndim == 2:
            self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)
        elif ndim == 3:
            self.proj = nn.Conv3d(input_dim, output_dim, kernel_size=1, bias=bias)
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            init_xavier_uniform_zero_bias(self.proj)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.proj(x)
