import torch
import torch.distributed as dist
from torch import nn


class AsyncBatchNorm(nn.Module):
    def __init__(self, dim, momentum=0.9, eps=1e-6):
        super().__init__()
        assert 0. <= momentum <= 1.
        assert 0 < dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(1, dim))
        self.register_buffer("running_var", torch.ones(1, dim))
        self.async_handle = None

    def _update_running_stats(self, mean_var=None):
        # get current stats from async handles
        if mean_var is None:
            if self.async_handle is not None:
                # get from async handle
                x = self.async_handle.get_future().value()
                x = torch.concat(x)
                self.async_handle = None
                cur_mean = x.mean(dim=0, keepdim=True)
                cur_var = x.var(dim=0, keepdim=True)
                mean_var = (cur_mean, cur_var)

        # update running stats
        if mean_var is not None:
            cur_mean, cur_var = mean_var
            self.running_mean.mul_(self.momentum).add_(cur_mean, alpha=1. - self.momentum)
            print(f"mean: {self.running_mean.tolist()}")
            self.running_var.mul_(self.momentum).add_(cur_var, alpha=1. - self.momentum)
            print(f"var: {self.running_var.tolist()}")

    def forward(self, x):
        if not self.training:
            # update running stats for last train iteration
            self._update_running_stats()
            return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        if dist.is_initialized():
            # update running stats for previous iteration
            self._update_running_stats()
            # queue synchonization for current iteration
            tensor_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            self.async_handle = dist.all_gather(tensor_list, x, async_op=True)
            return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            result = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            # update running stats directly
            cur_mean = x.mean(dim=0, keepdim=True)
            cur_var = x.var(dim=0, keepdim=True)
            self._update_running_stats((cur_mean, cur_var))
            return result
