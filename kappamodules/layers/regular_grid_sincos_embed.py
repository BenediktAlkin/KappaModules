import torch
from torch import nn

from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens, interpolate_sincos


class RegularGridSincosEmbed(nn.Module):
    def __init__(self, seqlens, dim: int, is_learnable: bool = False):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.is_learnable = is_learnable
        self.allow_interpolation = allow_interpolation
        if is_learnable:
            self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        else:
            self.register_buffer("embed", get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim).unsqueeze(0))
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_learnable:
            nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self):
        return self.embed
