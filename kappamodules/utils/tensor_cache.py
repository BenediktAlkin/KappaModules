from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def zeros(size, device=None, dtype=None):
    return torch.zeros(size, device=device, dtype=dtype)


@lru_cache(maxsize=None)
def ones(size, device=None, dtype=None):
    return torch.ones(size, device=device, dtype=dtype)


@lru_cache(maxsize=None)
def full(size, fill_value, device=None, dtype=None):
    return torch.full(size=size, fill_value=fill_value, device=device, dtype=dtype)


@lru_cache(maxsize=None)
def arange(start, end=None, step=1, dtype=None, device=None):
    if end is None:
        return torch.arange(start, dtype=dtype, device=device)
    return torch.arange(start, end, step, dtype=dtype, device=device)
