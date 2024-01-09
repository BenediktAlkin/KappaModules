from functools import partial

from torch import nn

from kappamodules.attention import PerceiverAttention1d
from kappamodules.init import init_norms_as_noaffine
from kappamodules.layers import DropPath
from kappamodules.modulation import Dit
from kappamodules.modulation.functional import modulate_scale_shift, modulate_gate
from .mlp import Mlp


class DitPerceiverBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_hidden_dim=None,
            cond_dim=None,
            drop_path=0.,
            bias=True,
            concat_query_to_kv=False,
            act_ctor=nn.GELU,
            eps=1e-6,
            init_weights="xavier_uniform",
            init_norms="nonaffine",
    ):
        super().__init__()
        norm_ctor = partial(nn.LayerNorm, elementwise_affine=False)
        self.init_norms = init_norms
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        cond_dim = cond_dim or dim
        # modulation
        self.modulation = Dit(cond_dim=cond_dim, out_dim=dim, num_outputs=8, init_weights=init_weights)
        #
        self.norm1q = norm_ctor(dim, eps=eps)
        self.norm1kv = norm_ctor(dim, eps=eps)
        self.attn = PerceiverAttention1d(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            concat_query_to_kv=concat_query_to_kv,
            init_weights=init_weights,
        )
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            bias=bias,
            act_ctor=act_ctor,
            init_weights=init_weights,
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

    def _attn_residual_path(self, q, kv, q_scale, q_shift, kv_scale, kv_shift, gate, attn_mask):
        q = modulate_scale_shift(self.norm1q(q), scale=q_scale, shift=q_shift)
        kv = modulate_scale_shift(self.norm1kv(kv), scale=kv_scale, shift=kv_shift)
        x = self.attn(q=q, kv=kv, attn_mask=attn_mask)
        return modulate_gate(x, gate=gate)

    def _mlp_residual_path(self, x, scale, shift, gate):
        return modulate_gate(self.mlp(modulate_scale_shift(self.norm2(x), scale=scale, shift=shift)), gate=gate)

    def forward(self, q, kv, cond, attn_mask=None):
        q_scale, q_shift, kv_scale, kv_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = self.modulation(cond)
        q = self.drop_path1(
            q,
            partial(
                self._attn_residual_path,
                kv=kv,
                q_scale=q_scale,
                q_shift=q_shift,
                kv_scale=kv_scale,
                kv_shift=kv_shift,
                gate=attn_gate,
                attn_mask=attn_mask,
            ),
        )
        q = self.drop_path2(q, partial(self._mlp_residual_path, scale=mlp_scale, shift=mlp_shift, gate=mlp_gate))
        return q
