import torch
import torch.nn as nn

class DropPath(nn.Sequential):
    """
    Efficiently drop paths (Stochastic Depth) per sample such that dropped samples are not processed.
    This is a subclass of nn.Sequential and can be used either as standalone Module or like nn.Sequential.
    Examples::
        >>> # use as nn.Sequential module
        >>> sequential_droppath = DropPath(nn.Linear(4, 4), drop_prob=0.2)
        >>> y = sequential_droppath(torch.randn(10, 4))

        >>> # use as standalone module
        >>> standalone_layer = nn.Linear(4, 4)
        >>> standalone_droppath = DropPath(drop_prob=0.2)
        >>> y = standalone_droppath(torch.randn(10, 4), standalone_layer)
    """

    def __init__(self, *args, drop_prob: float = 0., scale_by_keep: bool = True, stochastic_drop_prob: bool = False):
        super().__init__(*args)
        assert 0. <= drop_prob < 1.
        self.drop_prob = drop_prob
        self.keep_prob = 1. - drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob

    def forward(self, x, residual_path=None):
        assert (len(self) == 0) ^ (residual_path is None)
        if self.drop_prob == 0. or not self.training:
            if residual_path is None:
                return x + super().forward(x)
            else:
                return x + residual_path(x)
        # generate indices to keep (propagated through transform path)
        bs = len(x)
        if self.stochastic_drop_prob:
            perm = torch.empty(bs, device=x.device).bernoulli_(self.keep_prob).nonzero().squeeze(1)
            scale = 1 / self.keep_prob
        else:
            keep_count = max(int(bs * self.keep_prob), 1)
            scale = bs / keep_count
            perm = torch.randperm(bs, device=x.device)[:keep_count].sort().values

        # propagate
        if residual_path is None:
            y = super().forward(x[perm])
        else:
            y = residual_path(x[perm])
        if self.scale_by_keep:
            y = y * scale

        # merge drop (residual path) and keep (transform path)
        x2 = x[perm] + y
        mask = torch.zeros(len(x), *[1] * (x.ndim - 1), device=x.device, dtype=torch.bool)
        mask[perm] = True
        x = x.masked_scatter(mask, x2)
        return x

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'