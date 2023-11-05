import torch
from torch import nn


def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class OriginalWindowAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size=7,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = dim // num_heads
        attn_dim = head_dim * num_heads

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.rel_pos_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("rel_pos_index", get_relative_position_index(win_h, win_w))

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.rel_pos_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.rel_pos_bias_table[
            self.rel_pos_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_mask = self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
            attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
