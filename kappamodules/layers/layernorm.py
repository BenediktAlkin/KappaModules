import einops
from torch import nn


class LayerNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "batch_size dim height -> batch_size height dim")
        x = self.layer(x)
        x = einops.rearrange(x, "batch_size height dim -> batch_size dim height")
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "batch_size dim height width -> batch_size height width dim")
        x = self.layer(x)
        x = einops.rearrange(x, "batch_size height width dim -> batch_size dim height width")
        return x


class LayerNorm3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "batch_size dim height width depth -> batch_size height width depth dim")
        x = self.layer(x)
        x = einops.rearrange(x, "batch_size height width depth dim -> batch_size dim height width depth")
        return x
