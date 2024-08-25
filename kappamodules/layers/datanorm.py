import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DataNormStateDictPreHook:
    def __call__(self, module, *args, **kwargs):
        if dist.is_initialized():
            module.finish()


class DataNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, channel_first=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.channel_first = channel_first
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0.))
        self.mean_buffer = []
        self.var_buffer = []
        self._async_handle = None
        self.register_state_dict_pre_hook(DataNormStateDictPreHook())

    def _x_to_stats(self, x):
        # handles all use-cases
        # channel_first
        # - x.ndim == 2 (dim,) -> average over dim=[0]
        # - x.ndim == 3 (dim, height) -> average over dim=[0, 2]
        # - x.ndim == 4 (dim, height, width) -> average over dim=[0, 2, 3]
        # - x.ndim == 5 (dim, height, width, depth) -> average over dim=[0, 2, 3, 4]
        # channel_last
        # - x.ndim == 2 (dim,) -> average over dim=[0]
        # - x.ndim == 3 (height, dim) -> average over dim=[0, 1]
        # - x.ndim == 4 (height, width, dim) -> average over dim=[0, 1, 2]
        # - x.ndim == 5 (height, width, depth, dim) -> average over dim=[0, 1, 2, 3]
        if self.channel_first:
            x = einops.rearrange(x, "bs dim ... -> (bs ...) dim")
        else:
            x = einops.rearrange(x, "bs ... dim -> (bs ...) dim")
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        return mean, var

    def finish(self):
        self._update_stats(inplace=True)

    def _update_stats(self, inplace):
        if self._async_handle is not None:
            # get from async handle
            self._async_handle.wait()
            x = self._async_handle.get_future().value()
            x = torch.concat(x)
            self._async_handle = None
            # add stats to buffer
            xmean, xvar = self._x_to_stats(x)
            self.mean_buffer.append(xmean)
            self.var_buffer.append(xvar)

        # update should be called on every update
        assert len(self.mean_buffer) == 1
        assert len(self.var_buffer) == 1

        # accumulate stats
        mean = self.mean_buffer[0]
        var = self.var_buffer[0]
        # clear buffers
        self.mean_buffer.clear()
        self.var_buffer.clear()
        # calculate momentum
        self.num_batches_tracked += 1
        momentum = 1 / self.num_batches_tracked
        # update stats
        if inplace:
            # if used in nograd environment -> inplace
            self.mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            self.var.mul_(1 - momentum).add_(var, alpha=momentum)
        else:
            # if used in grad environment -> old mean/var are required for backward
            self.mean = (self.mean * (1 - momentum)).add_(mean, alpha=momentum)
            self.var = (self.var * (1 - momentum)).add_(var, alpha=momentum)

    def forward(self, x):
        if self.training:
            if len(x) == 1:
                raise NotImplementedError("DataNorm batch_size=1 requires syncing features instead of stats")

        # multi GPU -> queue communication of batch stats
        if dist.is_initialized():
            # update stats for previous iteration
            if self._async_handle is not None:
                self._update_stats(inplace=True)
            # queue synchonization for current iteration
            if self.training:
                assert self._async_handle is None
                tensor_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
                self._async_handle = dist.all_gather(tensor_list, x, async_op=True)

        # normalize
        og_x = x
        if not self.channel_first:
            x = einops.rearrange(x, "bs ... dim -> bs dim ...")
        x = F.batch_norm(
            input=x,
            running_mean=self.mean,
            running_var=self.var,
            eps=self.eps,
            # avoid updating mean/var
            training=False,
        )
        if not self.channel_first:
            x = einops.rearrange(x, "bs dim ... -> bs ... dim")

        # single GPU -> directly update stats
        if self.training and not dist.is_initialized():
            with torch.no_grad():
                xmean, xvar = self._x_to_stats(og_x)
                self.mean_buffer.append(xmean)
                self.var_buffer.append(xvar)
            self._update_stats(inplace=not x.requires_grad)

        return x
