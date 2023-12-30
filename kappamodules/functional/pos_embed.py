import einops
import torch
import torch.nn.functional as F


def get_sincos_1d_from_seqlen(seqlen: int, dim: int, max_wavelength: int = 10000):
    grid = torch.arange(seqlen, dtype=torch.double)
    return get_sincos_1d_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


def get_sincos_1d_from_grid(grid, dim: int, max_wavelength: int = 10000):
    if dim % 2 == 0:
        padding = None
    else:
        padding = torch.zeros(*grid.shape, 1)
        dim -= 1
    # generate frequencies for sin/cos (e.g. dim=8 -> omega = [1.0, 0.1, 0.01, 0.001])
    omega = 1. / max_wavelength ** (torch.arange(0, dim, 2, dtype=torch.double) / dim)
    # create grid of frequencies with timesteps
    # Example seqlen=5 dim=8
    # [0, 0.0, 0.00, 0.000]
    # [1, 0.1, 0.01, 0.001]
    # [2, 0.2, 0.02, 0.002]
    # [3, 0.3, 0.03, 0.003]
    # [4, 0.4, 0.04, 0.004]
    # Note: supports cases where grid is more than 1d
    out = grid.unsqueeze(-1) @ omega.unsqueeze(0)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.concat([emb_sin, emb_cos], dim=-1).float()
    if padding is None:
        return emb
    else:
        return torch.concat([emb, padding], dim=-1)


def get_sincos_2d_from_seqlens(seqlens, dim: int, max_wavelength: int = 10000):
    seqlen_h, seqlen_w = seqlens
    grid_h = torch.arange(seqlen_h, dtype=torch.double)
    grid_w = torch.arange(seqlen_w, dtype=torch.double)
    grid = torch.meshgrid(grid_h, grid_w, indexing="xy")
    grid = torch.stack(grid).reshape(2, seqlen_h, seqlen_w)
    return get_2d_sincos_pos_embed_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


def get_2d_sincos_pos_embed_from_grid(grid, dim: int, max_wavelength: int = 10000):
    assert dim % 2 == 0
    grid_h, grid_w = grid
    emb_h = get_sincos_1d_from_grid(grid=grid_h, dim=dim // 2, max_wavelength=max_wavelength)
    emb_w = get_sincos_1d_from_grid(grid=grid_w, dim=dim // 2, max_wavelength=max_wavelength)
    return torch.concat([emb_h, emb_w], dim=-1)


def get_sincos_3d_from_seqlens(seqlens, dim: int, max_wavelength: int = 10000):
    seqlen_x, seqlen_y, seqlen_z = seqlens
    grid_x = torch.arange(seqlen_x, dtype=torch.double)
    grid_y = torch.arange(seqlen_y, dtype=torch.double)
    grid_z = torch.arange(seqlen_z, dtype=torch.double)
    grid = torch.meshgrid(grid_x, grid_y, grid_z, indexing="xy")
    grid = torch.stack(grid).reshape(3, seqlen_x, seqlen_y, seqlen_z)
    return get_3d_sincos_pos_embed_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


def get_3d_sincos_pos_embed_from_grid(grid, dim: int, max_wavelength: int = 10000):
    assert dim % 3 == 0
    grid_x, grid_y, grid_z = grid
    emb_x = get_sincos_1d_from_grid(grid=grid_x, dim=dim // 3, max_wavelength=max_wavelength)
    emb_y = get_sincos_1d_from_grid(grid=grid_y, dim=dim // 3, max_wavelength=max_wavelength)
    emb_z = get_sincos_1d_from_grid(grid=grid_z, dim=dim // 3, max_wavelength=max_wavelength)
    return torch.concat([emb_x, emb_y, emb_z], dim=-1)


def get_sincos_pos_embed_from_seqlens(seqlens, dim: int, max_wavelength: int = 10000, indexing="ij"):
    assert isinstance(seqlens, (tuple, list))
    grids = [torch.arange(seqlen, dtype=torch.double) for seqlen in seqlens]
    if indexing == "xy":
        grids = reversed(grids)
    grid = torch.stack(torch.meshgrid(*grids, indexing=indexing))
    return get_sincos_pos_embed_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


def get_sincos_pos_embed_from_grid(grid, dim: int, max_wavelength: int = 10000):
    ndim = grid.size(0)
    if dim % ndim == 0:
        padding = None
    else:
        padding_dim = dim % ndim
        padding = torch.zeros(*grid.shape[1:], padding_dim)
        dim -= padding_dim
    pos_embed = torch.concat(
        [
            get_sincos_1d_from_grid(grid=grid[i], dim=dim // ndim, max_wavelength=max_wavelength)
            for i in range(ndim)
        ],
        dim=-1,
    )
    if padding is None:
        return pos_embed
    else:
        return torch.concat([pos_embed, padding], dim=-1)


def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    embed = F.interpolate(
        einops.rearrange(embed, "1 ... dim -> 1 dim ..."),
        size=seqlens,
        mode=mode,
    )
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed
