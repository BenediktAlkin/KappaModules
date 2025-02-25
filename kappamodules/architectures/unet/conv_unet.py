import torch
import einops
from torch import nn


class PatchMergingNd(nn.Module):
    def __init__(self, dim: int, ndim: int):
        super().__init__()
        self.ndim = ndim
        self.dim = dim
        factor = 2 ** ndim
        self.norm = nn.LayerNorm(factor * dim, eps=1e-6)
        self.reduction = nn.Linear(factor * dim, 2 * dim, bias=False)

    def forward(self, x):
        assert x.ndim - 2 == self.ndim
        resolution = x.shape[2:]
        assert all(resolution[i] % 2 == 0 for i in range(self.ndim))
        seqlens = [resolution[i] // 2 for i in range(self.ndim)]
        # generate generic pattern for ndim
        # pattern for 2d is: "bs c (h ph) (w pw) -> bs h w (ph pw c)"
        # pattern for 3d is: "bs c (x px) (y py) (z pz) -> bs x y z (px py pz c)"
        from_pattern = "c " + " ".join([f"(seqlen{i} patchsize{i})" for i in range(self.ndim)])
        to_pattern1 = " ".join([f"seqlen{i}" for i in range(self.ndim)])
        to_pattern2 = " ".join([f"patchsize{i}" for i in range(self.ndim)]) + " c"
        kwargs = {f"seqlen{i}": seqlens[i] for i in range(self.ndim)}
        x = einops.rearrange(x, f"bs {from_pattern} -> bs {to_pattern1} ({to_pattern2})", **kwargs)

        x = self.norm(x)
        x = self.reduction(x)

        x = einops.rearrange(x, "b ... c -> b c ...")
        return x


class PatchSplittingNd(nn.Module):
    """Patch splitting layer."""

    def __init__(self, dim: int, ndim: int):
        super().__init__()
        assert ndim == 2
        self.dim = dim
        self.ndim = ndim
        assert dim % 2 == 0, f"dim ({dim}) should be divisible by 2."
        if ndim == 2:
            self.lin1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        else:
            raise NotImplementedError
        self.norm = nn.LayerNorm(dim // 2)
        self.lin2 = nn.Linear(dim // 2, dim // 2, bias=False)

    def forward(self, x):
        assert x.ndim - 2 == self.ndim
        resolution = x.shape[2:]
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
        x = einops.rearrange(x, f"bs ({from_pattern2}) {from_pattern1} -> bs {to_pattern} c", **kwargs)

        x = self.norm(x)
        x = self.lin2(x)

        x = einops.rearrange(x, "b ... c -> b c ...")
        return x

class Film(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.modulation = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x, condition):
        # add spatial dims to condition
        assert condition.ndim == 2
        unsqueeze_shape = (-1, *[1] * (x.ndim - 2), condition.shape[-1])
        condition = condition.view(*unsqueeze_shape)
        assert x.ndim == condition.ndim

        # scale + shift
        scale, shift = self.modulation(condition).chunk(chunks=2, dim=-1)
        x = x * scale + shift
        return x


class ConvNextBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            ndim: int,
            padding_mode: str = "zeros",
            kernel_size: int = 7,
            condition_dim: int = None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        if ndim == 2:
            conv_ctor = nn.Conv2d
        else:
            raise NotImplementedError
        self.dw_conv = conv_ctor(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            padding_mode=padding_mode,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=condition_dim is None)
        self.film = nn.Identity() if condition_dim is None else Film(condition_dim=condition_dim, feature_dim=dim)
        self.pw_conv = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, condition=None):
        skip = x
        x = self.dw_conv(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        x = self.norm(x)
        x = self.film(x, condition=condition)
        x = self.pw_conv(x)
        x = einops.rearrange(x, "b ... c -> b c ...")
        return skip + x

class Stage(nn.Module):
    def __init__(
            self,
            dim: int,
            mode: str,
            depth: int,
            ndim: int,
            padding_mode: str = "zeros",
            condition_dim: int = None,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.depth = depth
        self.ndim = ndim
        self.padding_mode = padding_mode
        self.condition_dim = condition_dim
        assert ndim == 2

        self.blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    dim=dim,
                    ndim=ndim,
                    padding_mode=padding_mode,
                    condition_dim=condition_dim,
                )
                for _ in range(depth)
            ]
        )
        if self.mode is None or self.mode == "none":
            self.resample = None
        elif self.mode == "downsample":
            self.resample = PatchMergingNd(dim, ndim=ndim)
        elif self.mode == "upsample":
            self.resample = PatchSplittingNd(dim, ndim=ndim)
        else:
            raise NotImplementedError

    def forward(self, x, condition=None):
        for block in self.blocks:
            x = block(x, condition=condition)
        if self.resample is None:
            return x, None
        scaled = self.resample(x)
        return scaled, x


class Unet(nn.Module):
    def __init__(
            self,
            dim: int,
            stage_depths,
            ndim: int,
            patch_size: int = None,
            input_dim: int = None,
            output_dim: int = None,
            condition_dim: int = None,
            padding_mode: str = "zeros",
    ):
        super().__init__()
        self.dim = dim
        self.stage_depths = stage_depths
        self.num_stages = len(stage_depths)
        self.ndim = ndim
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        if ndim == 2:
            conv_ctor = nn.Conv2d
        else:
            raise NotImplementedError

        if patch_size is None:
            if input_dim is None:
                self.embed = nn.Identity()
            else:
                self.embed = conv_ctor(input_dim, dim, kernel_size=1)
        else:
            self.embed = conv_ctor(input_dim, dim, kernel_size=patch_size, stride=patch_size)

        # encoder
        downsample = ["downsample"] * (self.num_stages - 1) + [None]
        dims = [dim * 2 ** i for i in range(self.num_stages)]
        self.encoder_stages = nn.ModuleList(
            [
                Stage(
                    dim=dims[i],
                    mode=downsample[i],
                    depth=stage_depths[i],
                    ndim=ndim,
                    padding_mode=padding_mode,
                    condition_dim=condition_dim,
                )
                for i in range(self.num_stages)
            ]
        )

        # decoder
        upsample = ["upsample"] * (self.num_stages - 1) + [None]
        dims = list(reversed(dims))
        self.decoder_stages = nn.ModuleList(
            [
                Stage(
                    dim=dims[i],
                    mode=upsample[i],
                    depth=stage_depths[i],
                    ndim=ndim,
                    padding_mode=padding_mode,
                    condition_dim=condition_dim,
                )
                for i in range(self.num_stages)
            ]
        )

        if patch_size is None:
            self.pred = nn.Identity()
        else:
            if output_dim is None:
                raise NotImplementedError
            else:
                self.pred = nn.Sequential(
                    nn.Upsample(scale_factor=patch_size),
                    conv_ctor(dim, output_dim, kernel_size=3, padding=1, padding_mode=padding_mode),
                )

    def forward(self, x, condition=None):
        x = self.embed(x)

        # encoder
        skips = []
        for i, encoder_stage in enumerate(self.encoder_stages):
            x, skip = encoder_stage(x, condition=condition)
            skips.append(skip)

        # decoder
        for decoder_stage in self.decoder_stages:
            skip = skips.pop(-1)
            if skip is not None:
                x = x + skip
            x, _ = decoder_stage(x, condition=condition)

        # pred
        x = self.pred(x)
        return x

def main():
    model = Unet(
        dim=8,
        stage_depths=[2, 2, 2, 2],
        ndim=2,
        patch_size=4,
        input_dim=2,
        output_dim=2,
        condition_dim=8,
        padding_mode="circular",
    )
    print(model)
    y = model(torch.randn(1, 2, 256, 256), condition=torch.randn(1, 8))
    print(y.shape)
    # for m in model.modules():
        # if isinstance(m, nn.Conv2d):
        #     if m.padding == (0, 0):
        #         continue
        #     print(f"ks={m.kernel_size} pad={m.padding} pad={m.padding_mode}")
    for name, p in model.named_parameters():
        print(f"{name}: {p.shape}")


if __name__ == '__main__':
    main()