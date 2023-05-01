import torch.nn as nn

class Residual(nn.Sequential):
    def forward(self, x):
        return x + super().forward(x)