import einops
import torch
import torch.nn.functional as F
from torch import nn

from kappamodules.init import (
    init_xavier_uniform_zero_bias,
    init_xavier_uniform_merged_linear,
    init_truncnormal_zero_bias,
)


class MMDiTDotProductAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            proj_bias=True,
            seqlens=None,
            channel_first=False,
            init_weights="truncnormal002",
            init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seqlens = seqlens
        self.channel_first = channel_first
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj1 = nn.Linear(dim, dim, bias=proj_bias)
        self.proj2 = nn.Linear(dim, dim, bias=proj_bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.qkv1, num_layers=3)
            init_xavier_uniform_merged_linear(self.qkv2, num_layers=3)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            nn.init.zeros_(self.proj1.weight)
            nn.init.zeros_(self.proj2.weight)

    @staticmethod
    def to_channel_last(x):
        return einops.rearrange(x, "b c l -> b l c")

    @staticmethod
    def to_channel_first(x):
        return einops.rearrange(x, "b l c -> b c l")

    def forward(self, x1, x2, attn_mask=None):
        if self.channel_first:
            x1 = self.to_channel_last(x1)
            x2 = self.to_channel_last(x2)
        seqlen1 = x1.size(1)
        seqlen2 = x2.size(1)

        q1, k1, v1 = einops.rearrange(
            self.qkv1(x1),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        q2, k2, v2 = einops.rearrange(
            self.qkv2(x2),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        # concat in sequence dimension
        q = torch.concat([q1, q2], dim=2)
        k = torch.concat([k1, k2], dim=2)
        v = torch.concat([v1, v2], dim=2)
        # attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x1, x2 = x.split([seqlen1, seqlen2], dim=1)
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        if self.channel_first:
            x1 = self.to_channel_first(x1)
            x2 = self.to_channel_first(x2)
        return x1, x2
