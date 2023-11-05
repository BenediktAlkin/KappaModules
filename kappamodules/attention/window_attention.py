import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """ adaption of timm.models.swin_transformer.WindowAttention with scaled_dot_product_attention """

    def __init__(self, dim, window_size, num_heads=8, qkv_bias=True):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert isinstance(window_size, tuple)
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.window_area = np.prod(window_size)

        # 2D case
        # rel_pos_bias_table_numel = (2 * window_size[0] - 1) *(2 * window_size[1] - 1)
        # nD case
        rel_pos_bias_table_numel = np.prod([2 * window_dim - 1 for window_dim in window_size])
        self.rel_pos_bias_table = nn.Parameter(torch.zeros(rel_pos_bias_table_numel, num_heads))
        self.register_buffer("rel_pos_index", self._get_relative_position_index(), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

    def _get_relative_position_index(self):
        meshgrid = torch.meshgrid([torch.arange(win_dim) for win_dim in self.window_size], indexing="ij")
        coords = torch.stack(meshgrid).flatten(start_dim=1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = einops.rearrange(relative_coords, "two seqlen1 seqlen2 -> seqlen1 seqlen2 two").contiguous()
        # shift to start from 0
        for i, win_dim in enumerate(self.window_size):
            relative_coords[:, :, i] += win_dim - 1
        for i in range(len(self.window_size) - 1):
            relative_coords[:, :, i] *= sum(self.window_size) - 1
        return relative_coords.sum(-1).flatten()

    def forward(self, x, attn_mask=None):
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        # compose mask
        rel_pos_bias = einops.rearrange(
            self.rel_pos_bias_table[self.rel_pos_index],
            "(window_area1 window_area2) num_heads -> 1 num_heads window_area1 window_area2",
            window_area1=self.window_area,
            window_area2=self.window_area,
        )
        if attn_mask is None:
            attn_mask = rel_pos_bias
        else:
            # expand mask
            partitioned_bs = len(x)
            num_windows = len(attn_mask)
            attn_mask = attn_mask.repeat(partitioned_bs // num_windows, 1, 1).unsqueeze(1)
            attn_mask = attn_mask + rel_pos_bias

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.dtype))

        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x
