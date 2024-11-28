import einops
import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from kappamodules.init import (
    init_xavier_uniform_zero_bias,
    init_xavier_uniform_merged_linear,
    init_truncnormal_zero_bias,
)
from kappamodules.utils.param_checking import to_ntuple


class LocalgridAttention2d(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=3,
            num_heads=8,
            qkv_bias=True,
            proj_bias=True,
            init_weights="truncnormal002",
            init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        kernel_size = to_ntuple(kernel_size, n=2)
        padding = [ksize // 2 for ksize in kernel_size]
        assert all(ksize % 2 == 1 for ksize in kernel_size)

        self.kernel_size = kernel_size
        self.padding = padding
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.relative_position_bias = nn.Parameter(torch.zeros(1, num_heads, 1, np.prod(kernel_size)))
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.relative_position_bias)
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
            # init_weights == "torch" has no zero bias init
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, x, attn_mask=None):
        # shapes
        batch_size, seqlen_h, seqlen_w, _ = x.shape
        padding_h, padding_w = self.padding
        kernel_h, kernel_w = self.kernel_size
        kernel_h_half = kernel_h // 2
        kernel_w_half = kernel_w // 2
        assert kernel_h % 2 == 1, "even kernel sizes not supported"
        assert kernel_w % 2 == 1, "even kernel sizes not supported"
        kv_seqlen = np.prod(self.kernel_size)

        # project to attention space
        qkv = self.qkv(x)

        # pad + attention mask
        assert attn_mask is None
        qkv = F.pad(qkv, pad=(0, 0, padding_w, padding_w, padding_h, padding_h), mode="constant", value=0)
        attn_mask = torch.zeros(size=(seqlen_h, seqlen_w), device=x.device)
        attn_mask = F.pad(
            attn_mask,
            pad=(padding_h, padding_h, padding_w, padding_w),
            mode="constant",
            value=float("-inf"),
        )

        # create index
        # TODO cache these tensor creations
        q_idx = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(padding_h, seqlen_h + padding_h, device=x.device),
                    torch.arange(padding_w, seqlen_w + padding_w, device=x.device),
                ],
                indexing="ij",
            ),
        )
        kv_idx = einops.repeat(
            q_idx,
            "ndim seqlen_h seqlen_w -> kv_seqlen ndim seqlen_h seqlen_w",
            kv_seqlen=kv_seqlen,
        ).clone()
        offset = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(-kernel_h_half, kernel_h_half + 1, device=x.device),
                    torch.arange(-kernel_w_half, kernel_w_half + 1, device=x.device),
                ],
                indexing="ij",
            ),
        )
        offset = einops.rearrange(offset, "two kernel_h_half kernel_w_half -> (kernel_h_half kernel_w_half) two 1 1")
        kv_idx += offset

        # flatten indices
        seqlen_w_padded = seqlen_w + 2 * padding_w
        q_idx = q_idx[1] + q_idx[0] * seqlen_w_padded
        q_idx = einops.rearrange(q_idx, "seqlen_h seqlen_w -> (seqlen_h seqlen_w)")
        kv_idx = kv_idx[:, 1] + kv_idx[:, 0] * seqlen_w_padded
        kv_idx = einops.rearrange(kv_idx, "kv_seqlen seqlen_h seqlen_w -> kv_seqlen (seqlen_h seqlen_w)")

        # split per head
        q, k, v = einops.rearrange(
            qkv,
            "bs seqlen_h seqlen_w (three num_heads head_dim) -> three bs num_heads (seqlen_h seqlen_w) head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        # gather q
        q_idx_expand = q_idx[None, None, :, None].expand(batch_size, self.num_heads, -1, self.head_dim)
        q = torch.gather(q, dim=2, index=q_idx_expand)
        q = einops.rearrange(q, "batch_size num_heads seqlen head_dim -> (batch_size seqlen) num_heads 1 head_dim")
        # gather kv
        kv_idx = einops.rearrange(kv_idx, "kv_seqlen seqlen -> (seqlen kv_seqlen)")
        kv_idx_expand = kv_idx[None, None, :, None].expand(batch_size, self.num_heads, -1, self.head_dim)
        k = torch.gather(k, dim=2, index=kv_idx_expand)
        v = torch.gather(v, dim=2, index=kv_idx_expand)
        k = einops.rearrange(
            k,
            "batch_size num_heads (seqlen kv_seqlen) head_dim -> (batch_size seqlen) num_heads kv_seqlen head_dim",
            kv_seqlen=kv_seqlen,
        )
        v = einops.rearrange(
            v,
            "batch_size num_heads (seqlen kv_seqlen) head_dim -> (batch_size seqlen) num_heads kv_seqlen head_dim",
            kv_seqlen=kv_seqlen,
        )
        # gather mask
        attn_mask = attn_mask.flatten()[kv_idx]
        attn_mask = einops.repeat(
            attn_mask,
            "(seqlen kv_seqlen) -> (batch_size seqlen) 1 1 kv_seqlen",
            batch_size=batch_size,
            kv_seqlen=kv_seqlen,
        )
        # add relative positional bias
        attn_mask = attn_mask + self.relative_position_bias

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = einops.rearrange(
            x,
            "(bs seqlen_h seqlen_w) num_heads 1 head_dim -> bs seqlen_h seqlen_w (num_heads head_dim)",
            bs=batch_size,
            seqlen_h=seqlen_h,
            seqlen_w=seqlen_w,
        )
        x = self.proj(x)
        return x
