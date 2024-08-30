import torch


def to_logscale(x):
    return torch.sign(x) * (torch.log1p(x.abs()))


def from_logscale(x):
    return torch.sign(x) * (x.abs().exp() - 1)
