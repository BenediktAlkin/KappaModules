import torch.nn.functional as F
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
            num_modalities=2,
            cond_dim=None,
            qkv_bias=True,
            act_ctor=nn.GELU,
            eps=1e-6,
            init_weights="truncnormal002",
            init_last_proj_zero=False,
            init_gate_zero=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.eps = eps
        # properties
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        cond_dim = cond_dim or dim
        # modulation
        self.modulation = Dit(
            cond_dim=cond_dim,
            out_dim=dim,
            init_weights=init_weights,
            num_outputs=6 * num_modalities,
            gate_indices=list(range(2, 6 * num_modalities, 3)),
            init_gate_zero=init_gate_zero,
        )
        # attn
        self.attn = MMDiTDotProductAttention(
            dim=dim,
            num_heads=num_heads,
            num_modalities=num_modalities,
            qkv_bias=qkv_bias,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        # mlp
        self.mlp = nn.ModuleList(
            [
                Mlp(
                    in_dim=dim,
                    hidden_dim=mlp_hidden_dim,
                    act_ctor=act_ctor,
                    init_weights=init_weights,
                    init_last_proj_zero=init_last_proj_zero,
                )
                for _ in range(num_modalities)
            ],
        )

    def _attn_residual_path(self, x, scales, shifts, gates, attn_mask):
        assert isinstance(x, list)
        assert isinstance(scales, list)
        assert isinstance(shifts, list)
        assert isinstance(gates, list)
        assert len(x) == len(scales)
        assert len(x) == len(shifts)
        assert len(x) == len(gates)
        for i in range(len(x)):
            x[i] = F.layer_norm(x[i], [self.dim], eps=self.eps)
            x[i] = modulate_scale_shift(x[i], scale=scales[i], shift=shifts[i])
        x = self.attn(*x, attn_mask=attn_mask)
        for i in range(len(x)):
            x[i] = modulate_gate(x[i], gate=gates[i])
        return x

    def _mlp_residual_path(self, x, scales, shifts, gates):
        assert isinstance(x, list)
        assert isinstance(scales, list)
        assert isinstance(shifts, list)
        assert isinstance(gates, list)
        assert len(x) == len(scales)
        assert len(x) == len(shifts)
        assert len(x) == len(gates)
        for i in range(len(x)):
            x[i] = F.layer_norm(x[i], [self.dim], eps=self.eps)
            x[i] = modulate_scale_shift(x[i], scale=scales[i], shift=shifts[i])
            x[i] = self.mlp[i](x[i])
            x[i] = modulate_gate(x[i], gate=gates[i])
        return x

    def forward(self, *args, cond, attn_mask=None):
        scales_shifts_gates = self.modulation(cond)
        scales = [scales_shifts_gates[i] for i in range(0, 3 * self.num_modalities, 3)]
        shifts = [scales_shifts_gates[i] for i in range(1, 3 * self.num_modalities, 3)]
        gates = [scales_shifts_gates[i] for i in range(2, 3 * self.num_modalities, 3)]
        x = list(args)
        x = self._attn_residual_path(
            x=x,
            scales=scales,
            shifts=shifts,
            gates=gates,
            attn_mask=attn_mask,
        )
        scales = [scales_shifts_gates[i] for i in range(0 + 3 * self.num_modalities, 6 * self.num_modalities, 3)]
        shifts = [scales_shifts_gates[i] for i in range(1 + 3 * self.num_modalities, 6 * self.num_modalities, 3)]
        gates = [scales_shifts_gates[i] for i in range(2 + 3 * self.num_modalities, 6 * self.num_modalities, 3)]
        x = self._mlp_residual_path(
            x=x,
            scales=scales,
            shifts=shifts,
            gates=gates,
        )
        return x
