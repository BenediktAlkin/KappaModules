import einops
from torch import nn

from kappamodules.init import init_xavier_uniform_zero_bias, init_xavier_uniform_merged_linear


class LinearAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.init_weights = init_weights
        self.channel_first = channel_first

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.qkv, num_layers=3)
        else:
            raise NotImplementedError

    def to_channel_last(self, x):
        raise NotImplementedError

    def to_channel_first(self, x, og_shape):
        raise NotImplementedError

    def forward(self, x):
        if self.channel_first:
            og_shape = x.shape
            x = self.to_channel_last(x)
        else:
            og_shape = None

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

        if self.channel_first:
            x = self.to_channel_first(x, og_shape=og_shape)
        return x


class LinearAttention1d(LinearAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c l -> b l c")

    def to_channel_first(self, x, og_shape):
        return einops.rearrange(x, "b l c -> b c l")


class LinearAttention2d(LinearAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c h w -> b (h w) c")

    def to_channel_first(self, x, og_shape):
        _, _, h, w = og_shape
        return einops.rearrange(x, "bs (h w) dim -> bs dim h w", h=h, w=w)


class LinearAttention3d(LinearAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c h w d -> b (h w d) c")

    def to_channel_first(self, x, og_shape):
        _, _, h, w, d = og_shape
        return einops.rearrange(x, "bs (h w d) dim -> bs dim h w d", h=h, w=w, d=d)
