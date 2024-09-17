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


def interpolate_sincos(embed, seqlens, mode: str = "bicubic", interpolate_offset: float = None):
    assert embed.ndim - 2 == len(seqlens)
    embed = einops.rearrange(embed, "1 ... dim -> 1 dim ...")
    if interpolate_offset:
        # legacy interpolation from DINO/DINOv2
        # there is quite a substantial numerical difference to the "cleaner" version
        scale_factor = [(seqlens[i] + interpolate_offset) / embed.size(i + 2) for i in range(len(seqlens))]
        embed = F.interpolate(embed, scale_factor=scale_factor, mode=mode)
    else:
        embed = F.interpolate(embed, size=seqlens, mode=mode)
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed


def relative_position_indices(seqlens, num_aux_tokens):
    """ creates a bias for each relative distance """
    assert len(seqlens) == 2
    assert num_aux_tokens == 1

    seqlen0, seqlen1 = seqlens
    # position to position bias: (2 * seqlen0 - 1) * (2 * seqlen1 - 1)
    # 3 interaction types with cls: cls to cls, cls to patch, patch to cls
    num_distinct_distances = (2 * seqlen0 - 1) * (2 * seqlen1 - 1) + 3

    # create indices (2, seqlen0, seqlen1)
    abs_coords = torch.stack(torch.meshgrid([torch.arange(seqlen0), torch.arange(seqlen1)], indexing="ij"))
    abs_coords_flat = einops.rearrange(abs_coords, "ndim ... -> ndim (...)")
    # abs to rel: (2, seqlen0 * seqlen1) -> (2, seqlen0 * seqlen1, seqlen0 * seqlen1)
    rel_coords = abs_coords_flat[:, :, None] - abs_coords_flat[:, None, :]
    rel_coords = einops.rearrange(rel_coords, "ndim ... -> ... ndim").contiguous()
    rel_coords[:, :, 0] += seqlen0 - 1  # shift to start from 0
    rel_coords[:, :, 1] += seqlen1 - 1
    rel_coords[:, :, 0] *= 2 * seqlen1 - 1

    # create indices for looking up positional bias from table
    rel_pos_index = rel_coords.new_zeros(size=(seqlen0 * seqlen1 + 1, seqlen0 * seqlen1 + 1))
    # patch to patch
    rel_pos_index[1:, 1:] = rel_coords.sum(-1)
    # cls to cls
    rel_pos_index[0, 0] = num_distinct_distances - 1
    # cls to patch
    rel_pos_index[0:, 0] = num_distinct_distances - 2
    # patch to cls
    rel_pos_index[0, 0:] = num_distinct_distances - 3

    return rel_pos_index, num_distinct_distances