import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1e-5):
        super().__init__()
        if init_scale is None:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.full(size=(dim,), fill_value=init_scale))

    def forward(self, x):
        if self.gamma is None:
            return x
        return x * self.gamma
