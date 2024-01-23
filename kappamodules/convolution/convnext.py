from functools import partial

import torch
from torch import nn

from kappamodules.init import init_truncnormal_zero_bias
from kappamodules.layers import LayerNorm3d, LayerNorm2d, LayerNorm1d
from kappamodules.mlp import Mlp
from kappamodules.layers import DropPath


class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0., conv_ctor=nn.Conv2d, norm_ctor=LayerNorm2d):
        super().__init__()
        self.drop_path = DropPath(drop_prob=drop_path)
        self.conv_dw = conv_ctor(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        if isinstance(self.conv_dw, nn.Conv1d):
            ndim = 1
        elif isinstance(self.conv_dw, nn.Conv2d):
            ndim = 2
        elif isinstance(self.conv_dw, nn.Conv3d):
            ndim = 3
        else:
            raise NotImplementedError
        self.norm = norm_ctor(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=dim * 4, ndim=ndim, use_global_response_norm=True)

    def _forward(self, x):
        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x

    def forward(self, x):
        return self.drop_path(x, self._forward)


class ConvNextStage(nn.Module):
    def __init__(self, input_dim, output_dim, depth, drop_path_rates=None, conv_ctor=nn.Conv2d, norm_ctor=LayerNorm2d):
        super().__init__()
        if input_dim != output_dim:
            self.downsampling = nn.Sequential(
                norm_ctor(input_dim),
                conv_ctor(input_dim, output_dim, kernel_size=2, stride=2),
            )
        else:
            self.downsampling = nn.Identity()
        drop_path_rates = drop_path_rates or [0. * depth]
        self.blocks = nn.Sequential(
            *[
                ConvNextBlock(
                    dim=output_dim,
                    conv_ctor=conv_ctor,
                    norm_ctor=norm_ctor,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ],
        )

    def forward(self, x):
        x = self.downsampling(x)
        x = self.blocks(x)
        return x


class ConvNext(nn.Module):
    """ minimal version of timm.models.convnext.ConvNext that works with 3d """

    def __init__(
            self,
            patch_size,
            input_dim,
            depths,
            dims,
            drop_path_rate=0.0,
            drop_path_decay=True,
            ndim=2,
            eps=1e-6,
    ):
        super().__init__()
        assert len(dims) == len(depths)

        # ctors
        if ndim == 1:
            conv_ctor = nn.Conv1d
            norm_ctor = partial(LayerNorm1d, eps=eps)
        elif ndim == 2:
            conv_ctor = nn.Conv2d
            norm_ctor = partial(LayerNorm2d, eps=eps)
        elif ndim == 3:
            conv_ctor = nn.Conv3d
            norm_ctor = partial(LayerNorm3d, eps=eps)
        else:
            raise NotImplementedError

        # stem
        self.stem = nn.Sequential(
            conv_ctor(input_dim, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_ctor(dims[0]),
        )
        # stages
        if drop_path_decay:
            dprs = [dpr.tolist() for dpr in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        else:
            dprs = [dpr.tolist() for dpr in torch.tensor(drop_path_rate).repeat(sum(depths)).split(depths)]
        self.stages = nn.ModuleList(
            [
                ConvNextStage(
                    input_dim=dims[max(0, i - 1)],
                    output_dim=dims[i],
                    depth=depths[i],
                    drop_path_rates=dprs[i],
                    conv_ctor=conv_ctor,
                    norm_ctor=norm_ctor,
                )
                for i in range(len(dims))
            ],
        )
        self.reset_parameters()

    def reset_parameters(self):
        init_truncnormal_zero_bias(self)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
