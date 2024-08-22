import einops
import torch
from torch import nn


class ContinuousPosEmbed(nn.Module):
    def __init__(
            self,
            dim: int,
            ndim: int,
            mode: str = "sincos",
            max_value=None,
            interpolation: str = "linear",
            max_wavelength: int = 10000,
            dtype=torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.max_value = max_value
        self.max_wavelength = max_wavelength
        self.mode = mode
        self.interpolation = interpolation

        # if dim is not cleanly divisible -> cut away trailing dimensions
        ndim_padding = dim % ndim
        dim_per_ndim = (dim - ndim_padding) // ndim
        if mode == "sincos":
            assert max_value is None, "max_value is defined, but mode is sincos -> use learnable mode"
            sincos_padding = dim_per_ndim % 2
            self.padding = ndim_padding + sincos_padding * ndim
            eff_dim_per_wave = (self.dim - self.padding) // ndim
            assert eff_dim_per_wave > 0
            self.register_buffer(
                "omega",
                1. / max_wavelength ** (torch.arange(0, eff_dim_per_wave, 2, dtype=dtype) / eff_dim_per_wave),
            )
            self.embed = None
        elif mode == "learnable":
            assert max_value is not None
            if ndim == 1 and not isinstance(max_value, (list, tuple)):
                max_value = [max_value]
            assert len(max_value) == ndim
            self.padding = ndim_padding
            self.embed = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(max_value[i], dim_per_ndim) * 0.02)
                    for i in range(ndim)
                ],
            )
            self.omega = None
        else:
            raise NotImplementedError

    def forward(self, coords):
        if self.mode == "sincos":
            # sine/cosine embedding
            out_dtype = coords.dtype
            ndim = coords.shape[-1]
            assert self.ndim == ndim
            out = coords.unsqueeze(-1).to(self.omega.dtype) @ self.omega.unsqueeze(0)
            emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
            if coords.ndim == 3:
                emb = einops.rearrange(emb, "bs num_points ndim dim -> bs num_points (ndim dim)")
            elif coords.ndim == 2:
                emb = einops.rearrange(emb, "num_points ndim dim -> num_points (ndim dim)")
            else:
                raise NotImplementedError
            emb = emb.to(out_dtype)
        elif self.mode == "learnable":
            if coords.ndim == 1:
                assert self.ndim == 1
                coords = coords.unsqueeze(1)
            assert coords.size(1) == self.ndim
            if coords.dtype == torch.long:
                # no interpolation needed
                emb = torch.concat([self.embed[i][coords[:, i]] for i in range(self.ndim)], dim=1)
            else:
                if self.interpolation == "linear":
                    floored = coords.floor()
                    ceiled = coords.ceil()
                    ceil_weight = coords - floored
                    floor_weight = 1 - ceil_weight
                    emb_floor = torch.concat(
                        [
                            self.embed[i][floored[:, i].long()] * floor_weight[:, i].unsqueeze(1)
                            for i in range(self.ndim)
                        ],
                        dim=1,
                    )
                    emb_ceil = torch.concat(
                        [
                            self.embed[i][ceiled[:, i].long()] * ceil_weight[:, i].unsqueeze(1)
                            for i in range(self.ndim)
                        ],
                        dim=1,
                    )
                    emb = emb_floor + emb_ceil
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.concat([emb, padding], dim=-1)
        return emb

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self).__name__}(dim={self.dim})"
