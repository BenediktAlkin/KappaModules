import torch.nn.functional as F
from torch import nn


class L1Loss(nn.Module):
    @staticmethod
    def forward(pred, target, reduction="mean"):
        return F.l1_loss(pred, target, reduction=reduction)
