import einops
import torch
import torch.nn.functional as F
from torch import nn

from kappamodules.init import (
    init_xavier_uniform_zero_bias,
    init_xavier_uniform_merged_linear,
    init_truncnormal_zero_bias,
)

class PerceiverAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, concat_query_to_kv=False, init_weights="truncnormal"):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.concat_query_to_kv = concat_query_to_kv
        self.init_weights = init_weights

        self.kv = nn.Linear(dim, dim * 2, bias=bias)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.kv, num_layers=2)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, q, kv, attn_mask=None):
        # project to attention space
        if self.concat_query_to_kv:
            kv = torch.concat([kv, q], dim=1)
        kv = self.kv(kv)
        q = self.q(q)

        # split per head
        q = einops.rearrange(
            q,
            "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x


class PerceiverAttention1d(PerceiverAttention):
    pass