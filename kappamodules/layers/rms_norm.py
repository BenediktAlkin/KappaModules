import torch
import torch.nn.functional as F
from torch import nn

from kappamodules.utils.shapes import to_ndim


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # keep 1d to remain consistent with excluding norm parameters from weight decay
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=1) * to_ndim(self.g.view(1, -1), ndim=x.ndim) * (x.shape[1] ** 0.5)
