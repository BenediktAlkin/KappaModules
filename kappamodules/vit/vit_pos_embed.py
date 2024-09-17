import torch
from torch import nn

from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens, interpolate_sincos


class VitPosEmbed(nn.Module):
    def __init__(
            self,
            seqlens,
            dim: int,
            is_learnable: bool = False,
            allow_interpolation: bool = True,
            interpolate_offset: float = None,
    ):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.is_learnable = is_learnable
        self.allow_interpolation = allow_interpolation
        self.interpolate_offset = interpolate_offset
        if is_learnable:
            self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        else:
            self.register_buffer("embed", get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim).unsqueeze(0))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        if self.is_learnable:
            nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation
            embed = interpolate_sincos(
                embed=self.embed,
                seqlens=x.shape[1:-1],
                interpolate_offset=self.interpolate_offset,
            )
        else:
            embed = self.embed
        return x + embed


# LEGACY remove
class VitPosEmbedNd(VitPosEmbed):
    pass


class VitPosEmbed1d(VitPosEmbed):
    def __init__(self, seqlens, *args, **kwargs):
        assert len(seqlens) == 1
        super().__init__(seqlens=seqlens, *args, **kwargs)


class VitPosEmbed2d(VitPosEmbed):
    def __init__(self, seqlens, *args, **kwargs):
        assert len(seqlens) == 2
        super().__init__(seqlens=seqlens, *args, **kwargs)


class VitPosEmbed3d(VitPosEmbed):
    def __init__(self, seqlens, *args, **kwargs):
        assert len(seqlens) == 3
        super().__init__(seqlens=seqlens, *args, **kwargs)
