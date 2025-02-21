import einops
from torch import nn


class PatchMergingNd(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, ndim: int, channel_first: bool):
        super().__init__()
        self.ndim = ndim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_first = channel_first
        factor = 2 ** ndim
        self.norm = nn.LayerNorm(factor * input_dim, eps=1e-6)
        self.reduction = nn.Linear(factor * input_dim, output_dim, bias=False)

    def forward(self, x):
        assert x.ndim - 2 == self.ndim
        if self.channel_first:
            resolution = x.shape[2:]
        else:
            resolution = x.shape[1:-1]
        assert all(resolution[i] % 2 == 0 for i in range(self.ndim))
        seqlens = [resolution[i] // 2 for i in range(self.ndim)]
        # generate generic pattern for ndim
        # pattern for 2d is: "bs c (h ph) (w pw) -> bs h w (ph pw c)"
        # pattern for 3d is: "bs c (x px) (y py) (z pz) -> bs x y z (px py pz c)"
        from_pattern = " ".join([f"(seqlen{i} patchsize{i})" for i in range(self.ndim)])
        if self.channel_first:
            from_pattern = f"c {from_pattern}"
        else:
            from_pattern = f"{from_pattern} c"
        to_pattern1 = " ".join([f"seqlen{i}" for i in range(self.ndim)])
        to_pattern2 = " ".join([f"patchsize{i}" for i in range(self.ndim)]) + " c"
        kwargs = {f"seqlen{i}": seqlens[i] for i in range(self.ndim)}
        x = einops.rearrange(x, f"bs {from_pattern} -> bs {to_pattern1} ({to_pattern2})", **kwargs)

        x = self.norm(x)
        x = self.reduction(x)

        if self.channel_first:
            x = einops.rearrange(x, "b ... c -> b c ...")
        return x
