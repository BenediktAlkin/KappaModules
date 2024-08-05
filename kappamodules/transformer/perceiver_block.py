from functools import partial

from torch import nn

from kappamodules.attention import PerceiverAttention1d
from kappamodules.init import init_norms_as_noaffine
from kappamodules.layers import DropPath
from .mlp import Mlp


class PerceiverBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            kv_dim=None,
            mlp_hidden_dim=None,
            drop_path=0.,
            act_ctor=nn.GELU,
            norm_ctor=nn.LayerNorm,
            bias=True,
            concat_query_to_kv=False,
            eps=1e-6,
            init_weights="xavier_uniform",
            init_norms="nonaffine",
            init_last_proj_zero=False,
    ):
        super().__init__()
        self.init_norms = init_norms
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        self.norm1q = norm_ctor(dim, eps=eps)
        self.norm1kv = norm_ctor(kv_dim or dim, eps=eps)
        self.attn = PerceiverAttention1d(
            dim=dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
            bias=bias,
            concat_query_to_kv=concat_query_to_kv,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            bias=bias,
            act_ctor=act_ctor,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.drop_path2 = DropPath(drop_prob=drop_path)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_norms == "torch":
            pass
        elif self.init_norms == "nonaffine":
            init_norms_as_noaffine(self.norm1q)
            init_norms_as_noaffine(self.norm1kv)
            init_norms_as_noaffine(self.norm2)
        else:
            raise NotImplementedError

    def _attn_residual_path(self, q, kv, attn_mask):
        return self.attn(q=self.norm1q(q), kv=self.norm1kv(kv), attn_mask=attn_mask)

    def _mlp_residual_path(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, q, kv, attn_mask=None):
        q = self.drop_path1(
            q,
            residual_path=self._attn_residual_path,
            residual_path_kwargs=dict(kv=kv, attn_mask=attn_mask),
        )
        q = self.drop_path2(q, self._mlp_residual_path)
        return q
