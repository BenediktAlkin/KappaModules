import torch
import torch.nn.functional as F
from torch import nn


class LearnedBatchNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.dim = dim
        self.affine = affine
        self.mean = nn.Parameter(torch.zeros(*self._shape()))
        self.logvar = nn.Parameter(torch.zeros(*self._shape()))
        if affine:
            self.weight = nn.Parameters(torch.ones(*self._shape()))
            self.bias = nn.Parameters(torch.zeros(*self._shape()))
        else:
            self.weight = None
            self.bias = None

    def _shape(self):
        return 1, self.dim

    def forward(self, x):
        return F.batch_norm(
            input=x,
            running_mean=self.mean,
            running_var=self.logvar.exp(),
            weight=self.weight,
            bias=self.bias,
            # avoid updating mean/var
            training=False,
        )


class LearnedBatchNorm1d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1


class LearnedBatchNorm2d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1, 1


class LearnedBatchNorm3d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1, 1, 1
