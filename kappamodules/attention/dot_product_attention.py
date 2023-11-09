import einops
import torch.nn.functional as F
from torch import nn


class DotProductAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _forward(self, x):
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x

    def forward(self, x):
        raise NotImplementedError


class DotProductAttention1d(DotProductAttention):
    def forward(self, x):
        return self._forward(x)


class DotProductAttention2d(DotProductAttention):
    def forward(self, x):
        _, _, h, w = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self._forward(x)
        x = einops.rearrange(x, "bs (h w) dim -> bs dim h w", h=h, w=w)
        return x


class DotProductAttention3d(DotProductAttention):
    def forward(self, x):
        _, _, h, w, d = x.shape
        x = einops.rearrange(x, "b c h w d -> b (h w d) c")
        x = self._forward(x)
        x = einops.rearrange(x, "bs (h w d) dim -> bs dim h w d", h=h, w=w, d=d)
        return x
