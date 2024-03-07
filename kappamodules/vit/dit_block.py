from functools import partial

from torch import nn

from kappamodules.attention import DotProductAttention1d, DotProductAttentionSlow
from kappamodules.init import init_norms_as_noaffine
from kappamodules.layers import DropPath
from kappamodules.modulation import Dit
from kappamodules.modulation.functional import modulate_scale_shift, modulate_gate
from .vit_mlp import VitMlp

# TODO remove (deprecated)
class DitBlock(nn.Module):
    """ adaptive norm block (https://github.com/facebookresearch/DiT) """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_hidden_dim=None,
            cond_dim=None,
            qkv_bias=True,
            drop_path=0.,
            use_flash_attention=True,
            eps=1e-6,
            init_weights="xavier_uniform",
            init_norms="nonaffine",
            init_gate_zero=False,
    ):
        super().__init__()
        # DiT uses non-affine LayerNorm and GELU with tanh-approximation
        norm_ctor = partial(nn.LayerNorm, elementwise_affine=False)
        act_ctor = partial(nn.GELU, approximate="tanh")
        # properties
        self.init_norms = init_norms
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        cond_dim = cond_dim or dim
        # modulation
        self.modulation = Dit(
            cond_dim=cond_dim,
            out_dim=dim,
            num_outputs=6,
            gate_indices=[2, 5],
            init_weights=init_weights,
            init_gate_zero=init_gate_zero,
        )
        # attn
        self.norm1 = norm_ctor(dim, eps=eps)
        attn_ctor = DotProductAttention1d if use_flash_attention else DotProductAttentionSlow
        self.attn = attn_ctor(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, init_weights=init_weights)
        self.drop_path1 = DropPath(drop_prob=drop_path)
        # mlp
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = VitMlp(in_dim=dim, hidden_dim=mlp_hidden_dim, act_ctor=act_ctor, init_weights=init_weights)
        self.drop_path2 = DropPath(drop_prob=drop_path)
        #
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_norms == "torch":
            pass
        elif self.init_norms == "nonaffine":
            init_norms_as_noaffine(self.norm1)
            init_norms_as_noaffine(self.norm2)
        else:
            raise NotImplementedError

    def _attn_residual_path(self, x, scale, shift, gate, attn_mask):
        x = modulate_scale_shift(self.norm1(x), scale=scale, shift=shift)
        x = self.attn(x, attn_mask=attn_mask)
        return modulate_gate(x, gate=gate)

    def _mlp_residual_path(self, x, scale, shift, gate):
        return modulate_gate(self.mlp(modulate_scale_shift(self.norm2(x), scale=scale, shift=shift)), gate=gate)

    def forward(self, x, cond, attn_mask=None):
        scale1, shift1, gate1, scale2, shift2, gate2 = self.modulation(cond)
        x = self.drop_path1(
            x,
            partial(
                self._attn_residual_path,
                scale=scale1,
                shift=shift1,
                gate=gate1,
                attn_mask=attn_mask,
            )
        )
        x = self.drop_path2(x, partial(self._mlp_residual_path, scale=scale2, shift=shift2, gate=gate2))
        return x
