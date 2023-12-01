from torch import nn
import torch
from kappamodules.functional.pos_embed import get_sincos_1d_from_grid
import einops

class ContinuousPosEmbed(nn.Module):
    def __init__(self, dim, ndim, max_wavelength: int = 10000, dtype=torch.double, output_dtype=torch.float32):
        super().__init__()
        assert dim % ndim == 0
        self.dim = dim
        self.ndim = ndim
        self.max_wavelength = max_wavelength
        self.dtype = dtype
        self.output_dtype = output_dtype
        dim_div_ndim = dim // ndim
        self.register_buffer(
            "omega",
            1. / max_wavelength ** (torch.arange(0, dim_div_ndim, 2, dtype=dtype) / dim_div_ndim),
        )

    def forward(self, coords):
        _, _, ndim = coords.shape
        assert self.ndim == ndim
        out = coords.unsqueeze(-1).to(self.dtype) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        emb = einops.rearrange(emb, "bs num_points ndim dim -> bs num_points (ndim dim)")
        return emb.to(self.output_dtype)
