import torch
from torch import nn


class VitClassTokens(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 1, init_std=0.02):
        super().__init__()
        self.num_tokens = num_tokens
        self.init_std = init_std
        if num_tokens > 0:
            self.tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
        else:
            self.tokens = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_tokens > 0:
            nn.init.normal_(self.tokens, std=self.init_std)

    def forward(self, x):
        if self.num_tokens == 0:
            return x
        assert x.ndim == 3
        tokens = self.tokens.expand(len(x), -1, -1)
        x = torch.concat([tokens, x], dim=1)
        return x

    def split(self, x):
        if self.num_tokens == 0:
            return None, x
        assert x.ndim == 3
        cls_tokens = x[:, :self.num_tokens]
        patch_tokens = x[:, self.num_tokens:]
        return cls_tokens, patch_tokens
