import einops
import torch.nn.functional as F
from torch import nn

from kappamodules.init import (
    init_xavier_uniform_zero_bias,
    init_xavier_uniform_merged_linear,
    init_truncnormal_zero_bias,
)


class LinformerAttention(nn.Module):
    """ DotProductAttention but with an additional projection layer before k and v that reduces sequence length """

    def __init__(
            self,
            dim,
            input_seqlen,
            kv_seqlen,
            num_heads=8,
            qkv_bias=True,
            channel_first=False,
            init_weights="truncnormal002",
            init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.input_seqlen = input_seqlen
        self.kv_seqlen = kv_seqlen
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.channel_first = channel_first
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_k_seqlen = nn.Linear(input_seqlen, kv_seqlen)
        self.to_v_seqlen = nn.Linear(input_seqlen, kv_seqlen)
        self.proj = nn.Linear(dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.qkv, num_layers=3)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            nn.init.zeros_(self.proj.weight)

    def to_channel_last(self, x):
        raise NotImplementedError

    def to_channel_first(self, x, og_shape):
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        assert attn_mask is None
        if self.channel_first:
            og_shape = x.shape
            x = self.to_channel_last(x)
        else:
            og_shape = None

        # to qkv
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs (num_heads head_dim) seqlen",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        # reduce kv seqlen
        k = self.to_k_seqlen(k)
        v = self.to_v_seqlen(v)
        # to_k_seqlen and to_v_seqlen expect seqlen at last position -> reshape
        q = einops.rearrange(
            q,
            "bs (num_heads head_dim) seqlen -> bs num_heads seqlen head_dim",
            num_heads=self.num_heads,
        )
        k = einops.rearrange(
            k,
            "bs (num_heads head_dim) seqlen -> bs num_heads seqlen head_dim",
            num_heads=self.num_heads,
        )
        v = einops.rearrange(
            v,
            "bs (num_heads head_dim) seqlen -> bs num_heads seqlen head_dim",
            num_heads=self.num_heads,
        )
        # attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        if self.channel_first:
            x = self.to_channel_first(x, og_shape=og_shape)
        return x


class LinformerAttention1d(LinformerAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c l -> b l c")

    def to_channel_first(self, x, og_shape):
        return einops.rearrange(x, "b l c -> b c l")


class LinformerAttention2d(LinformerAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c h w -> b (h w) c")

    def to_channel_first(self, x, og_shape):
        _, _, h, w = og_shape
        return einops.rearrange(x, "bs (h w) dim -> bs dim h w", h=h, w=w)


class LinformerAttention3d(LinformerAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c h w d -> b (h w d) c")

    def to_channel_first(self, x, og_shape):
        _, _, h, w, d = og_shape
        return einops.rearrange(x, "bs (h w d) dim -> bs dim h w d", h=h, w=w, d=d)
