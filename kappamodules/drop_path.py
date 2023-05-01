import torch
import torch.nn as nn

class DropPath(nn.Sequential):
    """ Efficiently drop paths (Stochastic Depth) per sample """

    def __init__(self, *args, drop_prob: float = 0., scale_by_keep: bool = True, stochastic_size: bool = False):
        super().__init__(*args)
        assert 0. <= drop_prob < 1.
        self.drop_prob = drop_prob
        self.keep_prob = 1. - drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_size = stochastic_size

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x + super().forward(x)
        # generate indices to keep (propagated through transform path)
        bs = len(x)
        if self.stochastic_size:
            perm = torch.empty(bs, device=x.device).bernoulli_(self.keep_prob).nonzero().squeeze(1)
            scale = 1 / self.keep_prob
        else:
            keep_count = max(int(bs * self.keep_prob), 1)
            scale = bs / keep_count
            perm = torch.randperm(bs, device=x.device)[:keep_count]

        # propagate
        y = super().forward(x[perm])
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