import torch
from torch import nn

class Stage(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode

    def forward(self, x):
        raise NotImplementedError