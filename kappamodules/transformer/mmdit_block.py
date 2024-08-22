from functools import partial

from torch import nn

from kappamodules.attention import MMDiTDotProductAttention
from kappamodules.modulation import Dit
from kappamodules.modulation.functional import modulate_scale_shift, modulate_gate
from .mlp import Mlp


class MMDitBlock(nn.Module):
    """ multi-modal adaptive norm block (https://arxiv.org/abs/2403.03206) """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_hidden_dim=None,
            cond_dim=None,
            qkv_bias=True,
            attn_ctor=MMDiTDotProductAttention,
            act_ctor=nn.GELU,
            eps=1e-6,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            init_gate_zero=False,
    ):
        super().__init__()
        # DiT uses non-affine LayerNorm and GELU with tanh-approximation
        norm_ctor = partial(nn.LayerNorm, elementwise_affine=False)
        # properties
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        cond_dim = cond_dim or dim
        # modulation
        self.modulation = Dit(
            cond_dim=cond_dim,
            out_dim=dim,
            init_weights=init_weights,
            num_outputs=12,
            gate_indices=[2, 5, 8, 11],
            init_gate_zero=init_gate_zero,
        )
        # attn
        self.norm1 = norm_ctor(dim, eps=eps)
        self.attn = attn_ctor(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        # mlp
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp1 = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            act_ctor=act_ctor,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.mlp2 = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            act_ctor=act_ctor,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )

    def _attn_residual_path(self, x1, x2, scale1, shift1, gate1, scale2, shift2, gate2, attn_mask):
        x1 = modulate_scale_shift(self.norm1(x1), scale=scale1, shift=shift1)
        x2 = modulate_scale_shift(self.norm1(x2), scale=scale2, shift=shift2)
        x1, x2 = self.attn(x1=x1, x2=x2, attn_mask=attn_mask)
        x1 = modulate_gate(x1, gate=gate1)
        x2 = modulate_gate(x2, gate=gate2)
        return x1, x2

    def _mlp_residual_path(self, x1, x2, scale1, shift1, gate1, scale2, shift2, gate2):
        x1 = modulate_gate(self.mlp1(modulate_scale_shift(self.norm2(x1), scale=scale1, shift=shift1)), gate=gate1)
        x2 = modulate_gate(self.mlp2(modulate_scale_shift(self.norm2(x2), scale=scale2, shift=shift2)), gate=gate2)
        return x1, x2

    def forward(self, x1, x2, cond, attn_mask=None):
        scale1, shift1, g1, scale2, shift2, g2, scale3, shift3, g3, scale4, shift4, g4 = self.modulation(cond)
        x1, x2 = self._attn_residual_path(
            x1=x1,
            x2=x2,
            scale1=scale1,
            shift1=shift1,
            gate1=g1,
            scale2=scale2,
            shift2=shift2,
            gate2=g2,
            attn_mask=attn_mask,
        )
        x1, x2 = self._mlp_residual_path(
            x1=x1,
            x2=x2,
            scale1=scale3,
            shift1=shift3,
            gate1=g3,
            scale2=scale4,
            shift2=shift4,
            gate2=g4,
        )
        return x1, x2
