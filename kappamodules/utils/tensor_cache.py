from functools import partial

import torch

_cache = {}


def _wrapper(key, ctor=None):
    if key in _cache:
        tensor = _cache[key]
        # noinspection PyProtectedMember
        assert tensor._version == 0, f"cached tensor with key={key} was modified inplace"
        return tensor
    assert ctor is not None
    tensor = ctor()
    _cache[key] = tensor
    return tensor


def named(name, ctor=None):
    # force string to avoid any possibility of named overwriting other attributes in the cache
    # e.g. by passing ("zeros", (5,), "cpu", torch.bool) as name it would overwrite a cached zeros tensor
    assert isinstance(name, str)
    return _wrapper(key=name, ctor=ctor)


def zeros(size, device=None, dtype=None):
    return _wrapper(
        key=("zeros", size, str(device), dtype),
        ctor=partial(torch.zeros, size=size, device=device, dtype=dtype),
    )


def zeros_like(tensor):
    return zeros(size=tensor.shape, device=tensor.device, dtype=tensor.dtype)


def ones(size, device=None, dtype=None):
    return _wrapper(
        key=("ones", size, str(device), dtype),
        ctor=partial(torch.ones, size=size, device=device, dtype=dtype),
    )


def ones_like(tensor):
    return ones(size=tensor.shape, device=tensor.device, dtype=tensor.dtype)


def full(size, fill_value, device=None, dtype=None):
    return _wrapper(
        key=("full", size, fill_value, str(device), dtype),
        ctor=partial(torch.full, size=size, fill_value=fill_value, device=device, dtype=dtype),
    )


def full_like(tensor, fill_value):
    return full(size=tensor.shape, fill_value=fill_value, device=tensor.device, dtype=tensor.dtype)


def arange(start, end=None, step=1, device=None, dtype=None):
    key = ("arange", start, end, step, str(device), dtype)
    if end is None:
        ctor = partial(torch.arange, start=0, end=start, step=step, device=device, dtype=dtype)
    else:
        ctor = partial(torch.arange, start=start, end=end, step=step, device=device, dtype=dtype)
    return _wrapper(key=key, ctor=ctor)
