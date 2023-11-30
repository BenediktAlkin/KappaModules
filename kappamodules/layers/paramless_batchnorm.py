from torch import nn


class ParamlessBatchNorm1d(nn.Module):
    """
    non-affine BatchNorm1d layer that doesn't need a dimension but also can't be used in eval mode
    this layer works with SyncBatchnorm
    """

    def __init__(self, num_features=None):
        super().__init__()
        self.norm = nn.BatchNorm1d(
            num_features=num_features,
            affine=False,
            track_running_stats=num_features is not None,
        )

    def forward(self, x):
        if not self.norm.track_running_stats:
            assert self.training
        return self.norm(x)
