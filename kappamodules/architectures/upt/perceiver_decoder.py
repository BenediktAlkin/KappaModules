import einops
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock
from torch import nn


class PerceiverDecoder(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            ndim,
            input_dim,
            output_dim,
            init_weights="truncnormal002",
            eps=1e-6,
    ):
        super().__init__()
        # create query
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.query = nn.Sequential(
            LinearProjection(dim, dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(dim, dim, init_weights=init_weights),
        )
        # perceiver
        self.proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)
        self.perc = PerceiverBlock(dim=dim, num_heads=num_heads, init_weights=init_weights, eps=eps)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

    def forward(self, x, pos, block_kwargs=None, unbatch_mask=None):
        if pos is None:
            assert not self.training, f"{type(self).__name__} expects query positions during training"
            return None
        assert x.ndim == 3
        assert pos.ndim == 3

        # create query
        query = self.query(self.pos_embed(pos))

        # project to perceiver dim
        x = self.proj(x)

        # perceiver
        x = self.perc(q=query, kv=x, **(block_kwargs or {}))

        # predict value
        x = self.norm(x)
        x = self.pred(x)

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        if len(pos) == 1:
            # batch_size=1 -> no padding is needed
            pass
        else:
            if unbatch_mask is not None:
                x = x[unbatch_mask]
        return x
