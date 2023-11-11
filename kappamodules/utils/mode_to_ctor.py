from torch import nn

from kappamodules.layers import Identity


def mode_to_norm_ctor(mode):
    if mode is None:
        return Identity, True
    mode = mode.lower().replace("_", "")
    if mode == "none":
        return Identity, True
    if mode in ["bn", "batchnorm", "batchnorm1d"]:
        return nn.BatchNorm1d, False
    if mode in ["batchnorm2d"]:
        return nn.BatchNorm2d, False
    if mode in ["batchnorm3d"]:
        return nn.BatchNorm3d, False
    if mode in ["ln", "layernorm"]:
        return nn.LayerNorm, True
    if mode in ["instancenorm1d"]:
        return nn.InstanceNorm1d, True
    if mode in ["instancenorm2d"]:
        return nn.InstanceNorm2d, True
    if mode in ["instancenorm3d"]:
        return nn.InstanceNorm3d, True
    if mode in ["gn", "groupnorm"]:
        return nn.GroupNorm, True
    if mode in ["lrn", "localresponsenorm"]:
        return nn.LocalResponseNorm, True
    raise NotImplementedError(f"no suitable norm constructor found for '{mode}'")
