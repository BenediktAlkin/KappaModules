import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from kappamodules.init import init_xavier_uniform_merged_linear
from kappamodules.layers import LinearProjection


class TranssolverAttention(nn.Module):
    """
    Adapted from https://github.com/thuml/Transolver/blob/main/Car-Design-ShapeNetCar/models/Transolver.py
    - readable reshaping operations via einops
    - merged qkv linear layer for higher GPU utilization
    - F.scaled_dot_product_attention instead of slow pytorch attention
    - possibility to mask tokens (required to process variable sized inputs)
    """

    def __init__(
            self,
            dim,
            num_heads,
            num_slices,
            dropout=0.0,
            qkv_bias=False,
            init_weights="truncnormal",
            init_last_proj_zero=False,
    ):
        super().__init__()
        dim_head = dim // num_heads
        self.dim_head = dim_head
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.dropout = dropout
        self.init_last_proj_zero = init_last_proj_zero
        self.temperature = nn.Parameter(torch.full(size=(1, num_heads, 1, 1), fill_value=0.5))

        self.in_project_x = LinearProjection(dim, dim, init_weights=init_weights)
        self.in_project_fx = LinearProjection(dim, dim, init_weights=init_weights)
        self.in_project_slice = LinearProjection(dim_head, num_slices, init_weights=init_weights)
        # this is contained in the original implementation but is useless as it is overwritten later on
        # when initializing all linear layers with truncnormal
        # for l in [self.in_project_slice]:
        #     # use a principled initialization
        #     torch.nn.init.orthogonal_(l.weight)
        self.qkv = LinearProjection(dim_head, dim_head * 3, bias=qkv_bias, init_weights=init_weights)
        self.proj = LinearProjection(dim, dim, init_weights=init_weights)
        self.proj_dropout = nn.Dropout(dropout)

        # init weights
        if init_weights == "xavier_uniform":
            init_xavier_uniform_merged_linear(self.qkv.proj, num_layers=3)
        if self.init_last_proj_zero:
            nn.init.zeros_(self.proj.proj.weight)
            # init_weights == "torch" has no zero bias init
            if self.proj.proj.bias is not None:
                nn.init.zeros_(self.proj.proj.bias)

    def forward(self, x, attn_mask=None):
        batch_size, seqlen, _ = x.shape

        # slice
        fx_mid = einops.rearrange(
            self.in_project_fx(x),
            "batch_size seqlen (num_heads dim_head) -> batch_size num_heads seqlen dim_head",
            num_heads=self.num_heads,
        ).contiguous()
        x_mid = einops.rearrange(
            self.in_project_x(x),
            "batch_size seqlen (num_heads dim_head) -> batch_size num_heads seqlen dim_head",
            num_heads=self.num_heads,
        ).contiguous()
        slice_weights = F.softmax(self.in_project_slice(x_mid) / self.temperature, dim=-1)
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, "only bool mask supported"
            assert attn_mask.ndim == 2
            assert len(attn_mask) == len(x)
            assert attn_mask.size(1) == seqlen
            attn_mask = einops.rearrange(attn_mask, "batch_size seqlen -> batch_size 1 seqlen 1").float()
            slice_weights = slice_weights * attn_mask
        slice_norm = einops.rearrange(
            slice_weights.sum(2),
            "batch_size num_heads num_slices -> batch_size num_heads num_slices 1",
        )
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) / (slice_norm + 1e-5)

        # attention among slice tokens
        q_slice_token, k_slice_token, v_slice_token = self.qkv(slice_token).chunk(3, dim=-1)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token,
            k_slice_token,
            v_slice_token,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = einops.rearrange(
            out_x,
            "batch_size num_heads seqlen dim_head -> batch_size seqlen (num_heads dim_head)",
        )
        out_x = self.proj(out_x)
        out_x = self.proj_dropout(out_x)
        return out_x
