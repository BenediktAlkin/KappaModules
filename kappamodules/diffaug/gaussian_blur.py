import einops
import torch
import torch.nn.functional as F
from torch import nn

# TODO test
def gaussian_blur(x, sigma):
    # apply stylegan3-like blur (clean version of https://github.com/NVlabs/stylegan3/blob/main/training/loss.py#L53)
    _, c, h, w = x.shape
    size = int(sigma * 3)
    f = torch.arange(-size, size + 1, device=x.device).div(sigma).square().neg().exp2()
    f /= f.sum()
    padding = f.size(0) // 2
    f = einops.repeat(f, "f -> c 1 f", c=c)
    # pad with zeros (edges will be biased towards zero)
    x = F.conv2d(input=x, weight=f.unsqueeze(2), padding=[0, padding], groups=c)
    x = F.conv2d(input=x, weight=f.unsqueeze(3), padding=[padding, 0], groups=c)
    return x


class GaussianBlur(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return gaussian_blur(x, self.sigma)