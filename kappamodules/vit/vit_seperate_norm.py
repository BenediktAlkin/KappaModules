import einops
import torch
from torch import nn


class VitSeperateNorm(nn.Module):
    def __init__(
            self,
            dim,
            num_aux_tokens=1,
            aux_norm=nn.LayerNorm,
            patch_norm=nn.LayerNorm,
            affine=True,
            eps=1e-6,
    ):
        assert num_aux_tokens > 0
        super().__init__()
        self.num_aux_tokens = num_aux_tokens
        self.aux_norm = self._instantiate_norm(dim=dim * num_aux_tokens, ctor=aux_norm, eps=eps, affine=affine)
        self.patch_norm = self._instantiate_norm(dim=dim, ctor=patch_norm, eps=eps, affine=affine)

    @staticmethod
    def _instantiate_norm(ctor, dim, affine, eps):
        if ctor == nn.LayerNorm:
            return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)
        if ctor == nn.BatchNorm1d:
            return nn.BatchNorm1d(dim, eps=eps, affine=affine)
        raise NotImplementedError

    def forward(self, x):
        if self.num_aux_tokens == 0:
            return self.patch_norm(x)

        # norm aux tokens (multiple token case: concat aux tokens -> batchnorm -> split)
        flat_aux_tokens = einops.rearrange(
            x[:, :self.num_aux_tokens],
            "batch_size num_aux_tokens dim -> batch_size (num_aux_tokens dim)",
        )
        aux_normed = einops.rearrange(
            self.aux_norm(flat_aux_tokens),
            "batch_size (num_aux_tokens dim) -> batch_size num_aux_tokens dim",
            num_aux_tokens=self.num_aux_tokens,
        )

        patch_tokens = x[:, self.num_aux_tokens:]
        patch_normed = self.patch_norm(patch_tokens)
        return torch.concat([aux_normed, patch_normed], dim=1)
