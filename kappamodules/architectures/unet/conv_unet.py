import torch
import einops
from torch import nn


class PatchMergingNd(nn.Module):
    def __init__(self, dim: int, ndim: int, channel_first: bool):
        super().__init__()
        self.ndim = ndim
        self.dim = dim
        self.channel_first = channel_first
        factor = 2 ** ndim
        self.norm = nn.LayerNorm(factor * dim, eps=1e-6)
        self.reduction = nn.Linear(factor * dim, 2 * dim, bias=False)

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


class PatchSplittingNd(nn.Module):
    """Patch splitting layer."""

    def __init__(self, dim: int, ndim: int, channel_first: bool):
        super().__init__()
        assert ndim == 2
        self.dim = dim
        self.ndim = ndim
        self.channel_first = channel_first
        assert dim % 2 == 0, f"dim ({dim}) should be divisible by 2."
        if channel_first:
            if ndim == 2:
                self.lin1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
            else:
                raise NotImplementedError
        else:
            self.lin1 = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)
        self.lin2 = nn.Linear(dim // 2, dim // 2, bias=False)

    def forward(self, x):
        assert x.ndim - 2 == self.ndim
        if self.channel_first:
            resolution = x.shape[2:]
        else:
            resolution = x.shape[1:-1]
        assert all(resolution[i] % 2 == 0 for i in range(self.ndim))
        # double channel dim
        x = self.lin1(x)
        # generate generic pattern for ndim
        # pattern for 2d is: "bs h w (ph pw c) -> bs (h ph) (w pw) c"
        # pattern for 3d is: "bs x y z (px py pz c) -> bs (x px) (y py) (z pz) c"
        to_pattern = " ".join([f"(seqlen{i} patchsize{i})" for i in range(self.ndim)])
        from_pattern1 = " ".join([f"seqlen{i}" for i in range(self.ndim)])
        from_pattern2 = " ".join([f"patchsize{i}" for i in range(self.ndim)]) + " c"
        kwargs = {f"patchsize{i}": 2 for i in range(self.ndim)}
        if self.channel_first:
            x = einops.rearrange(x, f"bs ({from_pattern2}) {from_pattern1} -> bs {to_pattern} c", **kwargs)
        else:
            x = einops.rearrange(x, f"bs {from_pattern1} ({from_pattern2}) -> bs {to_pattern} c", **kwargs)

        x = self.norm(x)
        x = self.lin2(x)

        if self.channel_first:
            x = einops.rearrange(x, "b ... c -> b c ...")
        return x


class Stage(nn.Module):
    def __init__(self, dim: int, mode: str, depth: int, ndim: int):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.depth = depth
        self.ndim = ndim
        assert ndim == 2

        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
                for _ in range(depth)
            ]
        )
        if self.mode is None or self.mode == "none":
            self.scale = None
        elif self.mode == "downsample":
            self.scale = PatchMergingNd(dim, ndim=2, channel_first=True)
        elif self.mode == "upsample":
            self.scale = PatchSplittingNd(dim, ndim=2, channel_first=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.scale is None:
            return x, None
        scaled = self.scale(x)
        return scaled, x


class Unet(nn.Module):
    def __init__(self, dim: int, depths):
        super().__init__()
        self.dim = dim
        self.depths = depths
        self.num_stages = len(depths)

        downsample = ["downsample"] * (self.num_stages - 1) + [None]
        dims = [dim * 2 ** i for i in range(self.num_stages)]
        self.encoder_stages = nn.ModuleList(
            [
                Stage(
                    dim=dims[i],
                    mode=downsample[i],
                    depth=depths[i],
                    ndim=2,
                )
                for i in range(self.num_stages)
            ]
        )
        upsample = ["upsample"] * (self.num_stages - 1) + [None]
        dims = list(reversed(dims))
        self.decoder_stages = nn.ModuleList(
            [
                Stage(
                    dim=dims[i],
                    mode=upsample[i],
                    depth=depths[i],
                    ndim=2,
                )
                for i in range(self.num_stages)
            ]
        )

    def forward(self, x):
        skips = []
        for i, encoder_stage in enumerate(self.encoder_stages):
            x, skip = encoder_stage(x)
            skips.append(skip)
        for decoder_stage in self.decoder_stages:
            x, _ = decoder_stage(x)
            skip = skips.pop(-1)
            if skip is not None:
                x = x + skip
        return x

def main():
    model = Unet(dim=8, depths=[2, 2])
    y = model(torch.randn(1, 8, 16, 16))
    print(y.shape)


if __name__ == '__main__':
    main()