def to_ndim(x, ndim):
    return x.reshape(*x.shape, *(1,) * (ndim - x.ndim))
