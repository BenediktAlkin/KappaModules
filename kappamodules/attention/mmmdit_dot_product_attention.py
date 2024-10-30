import einops
import torch
import torch.nn.functional as F
from torch import nn

from kappamodules.init import (
    init_xavier_uniform_zero_bias,
    init_xavier_uniform_merged_linear,
    init_truncnormal_zero_bias,
)
from kappamodules.utils.param_checking import to_ntuple

class MMMDiTDotProductAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_main_modalities,
            num_optional_modalities,
            num_heads=8,
            qkv_bias=True,
            proj_bias=True,
            channel_first=False,
            init_weights="truncnormal002",
            init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        num_modalities = num_main_modalities + num_optional_modalities
        dim = to_ntuple(dim, num_modalities)
        assert dim[0] % num_heads == 0, "dim[0] should be divisible by num_heads"
        # checks
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.num_main_modalities = num_main_modalities
        self.num_optional_modalities = num_optional_modalities
        self.channel_first = channel_first
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        # attention is done in main_branch dim
        attn_dim = dim[0]
        self.qkv = nn.ModuleList(
            [
                nn.Linear(dim[i], attn_dim * 3, bias=qkv_bias)
                for i in range(num_modalities)
            ],
        )
        self.proj = nn.ModuleList(
            [
                nn.Linear(attn_dim, dim[i], bias=proj_bias)
                for i in range(num_modalities)
            ],
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            for qkv in self.qkv:
                init_xavier_uniform_merged_linear(qkv, num_layers=3)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            for proj in self.proj:
                nn.init.zeros_(proj.weight)
                # init_weights == "torch" has no zero bias init
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

    @staticmethod
    def to_channel_last(x):
        return einops.rearrange(x, "b c l -> b l c")

    @staticmethod
    def to_channel_first(x):
        return einops.rearrange(x, "b l c -> b c l")

    def forward(self, *args, attn_mask=None):
        assert attn_mask is None
        x = list(args)
        assert len(x) == self.num_modalities
        assert all(x[i].ndim == 3 for i in range(self.num_modalities))
        if self.channel_first:
            for i in range(self.num_modalities):
                x[i] = self.to_channel_last(x[i])
        seqlens = [x[i].size(1) for i in range(self.num_modalities)]

        qs = []
        ks = []
        vs = []
        for i in range(self.num_modalities):
            q, k, v = einops.rearrange(
                self.qkv[i](x[i]),
                "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
                three=3,
                num_heads=self.num_heads,
            ).unbind(0)
            qs.append(q)
            ks.append(k)
            vs.append(v)
        # concat main in sequence dimension
        main_q = torch.concat(qs[:self.num_main_modalities], dim=2)
        main_k = torch.concat(ks[:self.num_main_modalities], dim=2)
        main_v = torch.concat(vs[:self.num_main_modalities], dim=2)
        # main attention
        main_x = F.scaled_dot_product_attention(main_q, main_k, main_v)
        main_x = einops.rearrange(main_x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        main_x = list(main_x.split(seqlens[:self.num_main_modalities], dim=1))

        # optional attention
        optional_x = []
        for i in range(self.num_main_modalities, self.num_modalities):
            q = qs[i]
            k = torch.concat([main_k.detach(), ks[i]], dim=2)
            v = torch.concat([main_v.detach(), vs[i]], dim=2)
            opt_x = F.scaled_dot_product_attention(q, k, v)
            opt_x = einops.rearrange(opt_x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
            optional_x.append(opt_x)

        # postprocess
        x = main_x + optional_x
        for i in range(self.num_modalities):
            x[i] = self.proj[i](x[i])
        if self.channel_first:
            for i in range(self.num_modalities):
                x[i] = self.to_channel_first(x[i])
        return x
