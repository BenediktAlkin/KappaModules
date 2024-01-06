from torch import nn


class IgnoreArgsAndKwargsWrapper(nn.Module):
    """
    wrapper for any nn.Module that ignores everything besides the first arg passed to forward
    use-case: pytorch_geometrics InstanceNorm
    self.message = km.Sequential(
      km.IgnoreArgsAndKwargsWrapper(nn.Linear(10, 10)),
      pyg.InstanceNorm(10),
      ...
    )
    y = self.message(x, batch=batch)
    """

    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x