import einops
from torch import nn


class VitBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        if x.ndim >= 3:
            shape = x.shape
            x = x.reshape(-1, shape[-1])
            x = self.batchnorm(x)
            x = x.reshape(*shape)
        elif x.ndim == 2:
            # support single token as well (e.g. if norm is applied after pooling)
            x = self.batchnorm(x)
        else:
            raise RuntimeError(f"expected >2d (batch_size, ..., dim) input but got {tuple(x.shape)}")
        return x
