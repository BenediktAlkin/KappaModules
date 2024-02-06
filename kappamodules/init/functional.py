from torch import nn

ALL_BATCHNORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
)

ALL_NORMS = (
    *ALL_BATCHNORMS,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LocalResponseNorm,
)

ALL_CONVS = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

ALL_LAYERS = (
    nn.Linear,
    *ALL_CONVS,
)


def init_with_scheme(module, scheme):
    if scheme == "torch":
        pass
    elif scheme in ["truncnormal", "truncnormal002"]:
        module.apply(init_truncnormal_zero_bias)
    elif scheme == "xavier_uniform":
        module.apply(init_xavier_uniform_zero_bias)
    else:
        raise NotImplementedError


def init_norm_as_noaffine(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.)


# LEGACY remove
def init_norms_as_noaffine(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.)


def init_layernorm_as_noaffine(m):
    if isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.)


def init_batchnorm_as_noaffine(m):
    if isinstance(m, ALL_BATCHNORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.)


def init_norm_as_identity(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 0.)


# LEGACY remove
def init_norms_as_identity(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 0.)


def init_layernorm_as_identity(m):
    if isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 0.)
    else:
        raise NotImplementedError


def init_batchnorm_as_identity(m):
    if isinstance(m, ALL_BATCHNORMS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        if m.weight is not None:
            nn.init.constant_(m.weight, 0.)
    else:
        raise NotImplementedError


def init_bias_to_zero(m):
    if isinstance(m, ALL_LAYERS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


def init_linear_bias_to_zero(m):
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


def init_conv_bias_to_zero(m):
    if isinstance(m, ALL_CONVS):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


def init_xavier_uniform_zero_bias(m, gain: float = 1.):
    if isinstance(m, ALL_LAYERS):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


def init_truncnormal_zero_bias(m, std=0.02):
    if isinstance(m, ALL_LAYERS):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


def init_linear_truncnormal_zero_bias(m, std=0.02):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


def init_xavier_uniform_merged_linear(module, num_layers):
    # https://github.com/facebookresearch/moco-v3/blob/main/vits.py#L35
    assert isinstance(module, nn.Linear)
    # treat the weights of the merged linear layer as individual layers
    # e.g. with attention num_layers=3 (considers Q, K, V separately)
    assert module.weight.shape[0] % num_layers == 0
    val = (6 / (module.weight.shape[0] // num_layers + module.weight.shape[1])) ** 0.5
    nn.init.uniform_(module.weight, -val, val)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.)
