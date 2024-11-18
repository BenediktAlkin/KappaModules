import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from kappamodules.layers import LinearProjection
from kappamodules.init import init_xavier_uniform_merged_linear

class TranssolverAttention(nn.Module):
    """
    Adapted from https://github.com/thuml/Transolver/blob/main/Car-Design-ShapeNetCar/models/Transolver.py
    - readable reshaping operations via einops
    - merged qkv linear layer for higher GPU utilization
    """

    def __init__(self, dim, num_heads, num_slices, dropout=0., qkv_bias=False, init_weights="truncnormal"):
        super().__init__()
        dim_head = dim // num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
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
        if init_weights == "xavier_uniform":
            init_xavier_uniform_merged_linear(self.qkv.proj, num_layers=3)
        self.to_out = nn.Sequential(
            LinearProjection(dim, dim, init_weights="truncnormal"),
            nn.Dropout(dropout)
        )

    def forward(self, x):
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
        slice_norm = einops.repeat(
            slice_weights.sum(2),
            "batch_size num_heads num_slices -> batch_size num_heads num_slices dim_head",
            dim_head=self.dim_head,
        )
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) / (slice_norm + 1e-5)

        # attention among slice tokens
        q_slice_token, k_slice_token, v_slice_token = self.qkv(slice_token).chunk(3, dim=-1)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        # deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = einops.rearrange(
            out_x,
            "batch_size num_heads seqlen dim_head -> batch_size seqlen (num_heads dim_head)",
        )
        return self.to_out(out_x)
