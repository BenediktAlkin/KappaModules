import einops
from torch import nn


class VitBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if x.ndim == 3:
            seqlen = x.size(1)
            x = einops.rearrange(x, "b l c -> (b l) c")
            x = super().forward(x)
            x = einops.rearrange(x, "(b l) c -> b l c", l=seqlen)
        elif x.ndim == 2:
            # support single token as well (e.g. if norm is applied after pooling)
            x = super().forward(x)
        else:
            raise RuntimeError(
                f"expected 3d (batch_size, seqlen, dim) or "
                f"2d (batch_size, dim) input got {tuple(x.shape)}"
            )
        return x


class VitBatchNorm2d(nn.BatchNorm1d):
    def forward(self, x):
        if x.ndim == 4:
            _, _, height, width = x.shape
            x = einops.rearrange(x, "b h w c -> (b h w) c")
            x = super().forward(x)
            x = einops.rearrange(x, "(b h w) c -> b h w c", l=seqlen)
        elif x.ndim == 2:
            # support single token as well (e.g. if norm is applied after pooling)
            x = super().forward(x)
        else:
            raise RuntimeError(
                f"expected 4d (batch_size, height, width, dim) or "
                f"2d (batch_size, dim) input got {tuple(x.shape)}"
            )
        return x


class VitBatchNorm3d(nn.BatchNorm1d):
    def forward(self, x):
        if x.ndim == 5:
            _, _, height, width = x.shape
            x = einops.rearrange(x, "b x y z c -> (b x y z) c")
            x = super().forward(x)
            x = einops.rearrange(x, "(b x y z) c -> b x y z c", l=seqlen)
        elif x.ndim == 2:
            # support single token as well (e.g. if norm is applied after pooling)
            x = super().forward(x)
        else:
            raise RuntimeError(
                f"expected 5d (batch_size, x, y, z, dim) or "
                f"2d (batch_size, dim) input got {tuple(x.shape)}"
            )
        return x
