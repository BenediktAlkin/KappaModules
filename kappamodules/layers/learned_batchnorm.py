import torch
from torch import nn


class LearnedBatchNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.dim = dim
        self.affine = affine
        self.mean = nn.Parameter(torch.zeros(*self._shape()))
        self.logvar = nn.Parameter(torch.zeros(*self._shape()))
        if affine:
            self.weight = nn.Parameter(torch.ones(*self._shape()))
            self.bias = nn.Parameter(torch.zeros(*self._shape()))
        else:
            self.weight = None
            self.bias = None

    def _shape(self):
        return 1, self.dim

    def forward(self, x):
        # NOTE: cant use F.batch_norm as it is not differentiable to running_mean/running_var
        return (x - self.mean) / self.logvar.exp() * self.weight + self.bias


class LearnedBatchNorm1d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1


class LearnedBatchNorm2d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1, 1


class LearnedBatchNorm3d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1, 1, 1
