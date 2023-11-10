from torch import nn

from kappamodules.attention.dot_product_attention import DotProductAttention1d
from kappamodules.attention.dot_product_attention_slow import DotProductAttentionSlow
from kappamodules.layers import DropPath
from .vit_mlp import VitMlp


class VitBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_hidden_dim=None,
            qkv_bias=True,
            drop_path=0.,
            act_ctor=nn.GELU,
            norm_ctor=nn.LayerNorm,
            use_flash_attention=True,
            eps=1e-6,
    ):
        super().__init__()
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        self.norm1 = norm_ctor(dim, eps=eps)
        attn_ctor = DotProductAttention1d if use_flash_attention else DotProductAttentionSlow
        self.attn = attn_ctor(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = VitMlp(in_dim=dim, hidden_dim=mlp_hidden_dim, act_ctor=act_ctor)
        self.drop_path2 = DropPath(drop_prob=drop_path)

    def _attn_residual_path(self, x):
        return self.attn(self.norm1(x))

    def _mlp_residual_path(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x):
        x = self.drop_path1(x, self._attn_residual_path)
        x = self.drop_path2(x, self._mlp_residual_path)
        return x
