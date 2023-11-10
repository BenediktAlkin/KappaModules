import einops
from torch import nn

from kappamodules.init import initialize_xavier_uniform_zero_bias, initialize_qkv_seperately


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def reset_parameters(self):
        self.apply(initialize_xavier_uniform_zero_bias)
        initialize_qkv_seperately(self)

    def _forward(self, x):
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        # attn
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        kv = k @ v.transpose(-2, -1)
        x = kv.transpose(-2, -1) @ q

        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x

    def forward(self, x):
        raise NotImplementedError


class LinearAttention1d(LinearAttention):
    def forward(self, x):
        return self._forward(x)


class LinearAttention2d(LinearAttention):
    def forward(self, x):
        _, _, h, w = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self._forward(x)
        x = einops.rearrange(x, "bs (h w) dim -> bs dim h w", h=h, w=w)
        return x


class LinearAttention3d(LinearAttention):
    def forward(self, x):
        _, _, h, w, d = x.shape
        x = einops.rearrange(x, "b c h w d -> b (h w d) c")
        x = self._forward(x)
        x = einops.rearrange(x, "bs (h w d) dim -> bs dim h w d", h=h, w=w, d=d)
        return x
