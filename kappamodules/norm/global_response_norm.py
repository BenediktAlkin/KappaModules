import torch
from torch import nn


class GlobalResponseNorm(nn.Module):
    """ adapted from timm.layers.grn.GlobalResponseNorm """

    def __init__(self, dim, eps=1e-6, ndim=None):
        super().__init__()
        self.eps = eps
        if ndim is None:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        elif ndim == 1:
            self.spatial_dim = (2,)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1)
        elif ndim == 2:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)
        elif ndim == 3:
            self.spatial_dim = (2, 3, 4)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1, 1)
        else:
            raise NotImplementedError

        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)


class GlobalResponseNorm1d(GlobalResponseNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=1)


class GlobalResponseNorm2d(GlobalResponseNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=2)


class GlobalResponseNorm3d(GlobalResponseNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=3)
