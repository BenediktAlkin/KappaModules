import einops
import torch
from torch import nn


class VitClassTokens(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 1, location="first", init_std=0.02):
        super().__init__()
        self.location = location
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
        if self.location == "first":
            x = torch.concat([tokens, x], dim=1)
        elif self.location == "middle":
            pre, post = x.chunk(chunks=2, dim=1)
            x = torch.concat([pre, tokens, post], dim=1)
        else:
            raise NotImplementedError
        return x

    def split(self, x):
        if self.num_tokens == 0:
            return None, x
        assert x.ndim == 3
        if self.location == "first":
            cls_tokens = x[:, :self.num_tokens]
            patch_tokens = x[:, self.num_tokens:]
        else:
            raise NotImplementedError
        return cls_tokens, patch_tokens

    def pool(self, x):
        if self.num_tokens == 0:
            raise NotImplementedError
        if self.location == "first":
            x = einops.rearrange(x[:, :self.num_tokens], "batchsize seqlen dim -> batchsize (seqlen dim)")
        elif self.location == "middle":
            middle = x.size(1) // 2
            half_num_tokens = self.num_tokens // 2
            start = middle - half_num_tokens
            end = start + self.num_tokens
            x = x[:, start:end]
        else:
            raise NotImplementedError
        return x
