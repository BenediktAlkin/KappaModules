from torch import nn


class ParamlessBatchNorm1d(nn.Module):
    """ non-affine BatchNorm1d layer that doesn't need a dimension but also can't be used in eval mode """

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=None, affine=False, track_running_stats=False)

    def forward(self, x):
        assert self.training
        return self.norm(x)
