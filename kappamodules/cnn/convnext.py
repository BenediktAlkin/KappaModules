import torch
from torch import nn
from kappamodules.vit import VitPatchEmbed
from functools import partial
from kappamodules.layers import LayerNorm3d, LayerNorm2d

class ConvNext(nn.Module):
    """ minimal version of timm.models.convnext.ConvNext that works with 3d """

    def __init__(
            self,
            patch_size,
            input_channels,
            depths,
            dims,
            ndim=2,
            eps=1e-6,
    ):
        super().__init__()
        if ndim == 2:
            conv_ctor = nn.Conv2d
            norm_ctor = partial(LayerNorm2d, eps=eps)
        elif ndim == 3:
            conv_ctor = nn.Conv3d
            norm_ctor = partial(LayerNorm3d, eps=eps)
        else:
            raise NotImplementedError

        self.stem = nn.Sequential(
            conv_ctor(input_channels, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_ctor(dims[0]),
        )