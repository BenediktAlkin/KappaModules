import torch.nn.functional as F
from torch import nn


class Normalize(nn.Module):
    def __init__(self, dim=1, p=2.0):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p)
