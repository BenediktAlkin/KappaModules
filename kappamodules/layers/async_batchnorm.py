import os

import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class AsyncBatchNormStateDictPreHook:
    def __call__(self, module, *args, **kwargs):
        if dist.is_initialized():
            module.finish()


class AsyncBatchNorm(nn.Module):
    def __init__(
            self,
            dim,
            momentum=0.9,
            affine=True,
            eps=1e-5,
            gradient_accumulation_steps=None,
            whiten=True,
            channel_first=True,
    ):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.affine = affine
        self.whiten = whiten
        self.eps = eps
        self.channel_first = channel_first
        self.gradient_accumulation_steps = gradient_accumulation_steps or os.environ.get("GRAD_ACC_STEPS", None) or 1
        assert self.gradient_accumulation_steps > 0
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.batchsize_buffer = []
        self.mean_buffer = []
        self.var_buffer = []
        self._async_handle = None
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.weight = None
            self.bias = None
        self.register_state_dict_pre_hook(AsyncBatchNormStateDictPreHook())

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
        if self.whiten:
            var = x.var(dim=0, unbiased=False)
        else:
            var = None
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
            self.batchsize_buffer.append(len(x))
            xmean, xvar = self._x_to_stats(x)
            self.mean_buffer.append(xmean)
            if xvar is not None:
                self.var_buffer.append(xvar)

        # check if update is already needed
        if len(self.mean_buffer) < self.gradient_accumulation_steps:
            return

        assert all(self.batchsize_buffer[0] == bsb for bsb in self.batchsize_buffer[1:])
        # accumulate stats
        if self.gradient_accumulation_steps == 1:
            mean = torch.stack(self.mean_buffer).mean(dim=0)
            if self.whiten:
                var = self.var_buffer[0]
            else:
                var = None
        else:
            # https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
            n = self.batchsize_buffer[0]
            xbar = self.mean_buffer[0]
            if self.whiten:
                sx = self.var_buffer[0]
            else:
                sx = None
            for i in range(1, self.gradient_accumulation_steps):
                m = self.batchsize_buffer[i]
                ybar = self.mean_buffer[i]
                if self.whiten:
                    sy = self.var_buffer[i]
                    sx = (
                            (n - 1) * sx + (m - 1) * sy / (n + m - 1)
                            + n * m * (xbar - ybar) ** 2 / (n + m) * (n + m - 1)
                    )
                xbar = (n * xbar + m * ybar) / (n + m)
                n += m
            mean = xbar
            var = sx
        # clear buffers
        self.batchsize_buffer.clear()
        self.mean_buffer.clear()
        self.var_buffer.clear()
        # update stats
        if inplace:
            # if used in nograd environment -> inplace
            self.mean.mul_(self.momentum).add_(mean, alpha=1. - self.momentum)
            if self.whiten:
                self.var.mul_(self.momentum).add_(var, alpha=1. - self.momentum)
        else:
            # if used in grad environment -> old mean/var are required for backward
            self.mean = (self.mean * self.momentum).add_(mean, alpha=1. - self.momentum)
            if self.whiten:
                self.var = (self.var * self.momentum).add_(var, alpha=1. - self.momentum)

    def forward(self, x):
        if self.training:
            if len(x) == 1:
                raise NotImplementedError("AsyncBatchNorm batch_size=1 requires syncing features instead of stats")

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
            weight=self.weight,
            bias=self.bias,
            eps=self.eps if self.whiten else 0,
            # avoid updating mean/var
            training=False,
        )
        if not self.channel_first:
            x = einops.rearrange(x, "bs dim ... -> bs ... dim")

        # single GPU -> directly update stats
        if self.training and not dist.is_initialized():
            with torch.no_grad():
                self.batchsize_buffer.append(len(og_x))
                xmean, xvar = self._x_to_stats(og_x)
                self.mean_buffer.append(xmean)
                if self.whiten:
                    self.var_buffer.append(xvar)
            if len(self.mean_buffer) == self.gradient_accumulation_steps:
                self._update_stats(inplace=not x.requires_grad)

        return x

    @classmethod
    def convert_async_batchnorm(cls, module):
        module_output = module
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module_output = AsyncBatchNorm(
                dim=module.num_features,
                momentum=module.momentum,
                affine=module.affine,
                eps=module.eps,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.mean = module.running_mean
            module_output.var = module.running_var
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_async_batchnorm(child))
        del module
        return module_output


class AsyncBatchNorm1d(AsyncBatchNorm):
    pass


class AsyncBatchNorm2d(AsyncBatchNorm):
    pass


class AsyncBatchNorm3d(AsyncBatchNorm):
    pass
