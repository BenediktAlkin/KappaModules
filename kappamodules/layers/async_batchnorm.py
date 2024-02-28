import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class AsyncBatchNormStateDictPreHook:
    def __call__(self, *args, **kwargs):
        if dist.is_initialized():
            raise NotImplementedError


class AsyncBatchNorm(nn.Module):
    def __init__(self, dim, momentum=0.9, affine=True, eps=1e-5, gradient_accumulation_steps=None):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.affine = affine
        self.eps = eps
        self.gradient_accumulation_steps = gradient_accumulation_steps or os.environ.get("GRAD_ACC_STEPS", None) or 1
        assert self.gradient_accumulation_steps > 0
        self.register_buffer("mean", torch.zeros(*self._shape()))
        self.register_buffer("var", torch.ones(*self._shape()))
        self.batchsize_buffer = []
        self.mean_buffer = []
        self.var_buffer = []
        self._async_handle = None
        if affine:
            self.weight = nn.Parameter(torch.ones(*self._shape()))
            self.bias = nn.Parameter(torch.zeros(*self._shape()))
        else:
            self.weight = None
            self.bias = None
        self.register_state_dict_pre_hook(AsyncBatchNormStateDictPreHook())

    def _shape(self):
        return self.dim,

    def _update_stats(self, inplace):
        assert all(self.batchsize_buffer[0] == bsb for bsb in self.batchsize_buffer[1:])
        # accumulate stats
        if self.gradient_accumulation_steps == 1:
            mean = torch.stack(self.mean_buffer).mean(dim=0)
            var = self.var_buffer[0]
        else:
            # https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
            sx = self.var_buffer[0]
            n = self.batchsize_buffer[0]
            xbar = self.mean_buffer[0]
            for i in range(1, self.gradient_accumulation_steps):
                m = self.batchsize_buffer[i]
                sy = self.var_buffer[i]
                ybar = self.mean_buffer[i]
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
            self.var.mul_(self.momentum).add_(var, alpha=1. - self.momentum)
        else:
            # if used in grad is required -> old mean/var are required for backward
            self.mean = (self.mean * self.momentum).add_(mean, alpha=1. - self.momentum)
            self.var = (self.var * self.momentum).add_(var, alpha=1. - self.momentum)

    def forward(self, x):
        if self.training:
            if len(x) == 1:
                raise NotImplementedError("AsyncBatchNorm with batch_size requires syncing features instead of stats")

        # multi GPU -> queue communication of batch stats
        if self.training and dist.is_initialized():
            assert x.requires_grad, "AsyncBatchNorm doesn't support no_grad in training mode"
            raise NotImplementedError

        og_x = x
        x = F.batch_norm(
            input=x,
            running_mean=self.mean,
            running_var=self.var,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            # avoid updating mean/var
            training=False,
        )

        # single GPU -> directly update stats
        if self.training and not dist.is_initialized():
            with torch.no_grad():
                self.batchsize_buffer.append(len(og_x))
                self.mean_buffer.append(og_x.mean(dim=0))
                self.var_buffer.append(og_x.var(dim=0, unbiased=False))
            if len(self.mean_buffer) == self.gradient_accumulation_steps:
                self._update_stats(inplace=not x.requires_grad)

        return x

    @classmethod
    def convert_async_batchnorm(cls, module):
        module_output = module
        if isinstance(module, nn.BatchNorm1d):
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
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_async_batchnorm(child))
        del module
        return module_output
