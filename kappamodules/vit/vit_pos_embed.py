import torch
from torch import nn

from kappamodules.functional.pos_embed import (
    get_sincos_1d_from_seqlen,
    get_sincos_2d_from_seqlens,
    get_sincos_3d_from_seqlens,
)


class VitPosEmbed1d(nn.Module):
    def __init__(self, seqlen: int, dim: int, is_learnable=False):
        super().__init__()
        self.seqlen = seqlen
        self.dim = dim
        self.is_learnable = is_learnable
        if is_learnable:
            self.embed = nn.Parameter(torch.zeros(1, seqlen, dim))
        else:
            self.register_buffer("embed", get_sincos_1d_from_seqlen(seqlen=seqlen, dim=dim).unsqueeze(0))
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_learnable:
            nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == 3
        return x + self.embed


class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens: int, dim: int, is_learnable=False):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.is_learnable = is_learnable
        if is_learnable:
            self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        else:
            self.register_buffer("embed", get_sincos_2d_from_seqlens(seqlens=seqlens, dim=dim).unsqueeze(0))
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_learnable:
            nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == 4
        return x + self.embed


class VitPosEmbed3d(nn.Module):
    def __init__(self, seqlens: int, dim: int, is_learnable=False):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.is_learnable = is_learnable
        if is_learnable:
            self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        else:
            self.register_buffer("embed", get_sincos_3d_from_seqlens(seqlens=seqlens, dim=dim).unsqueeze(0))
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_learnable:
            nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == 5
        return x + self.embed
