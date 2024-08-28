import einops
import torch
from torch import nn


class VitClassTokens(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 1, location="first", init_std=0.02, aggregate="flatten"):
        super().__init__()
        self.dim = dim
        self.location = location
        self.num_tokens = num_tokens
        self.init_std = init_std
        self.aggregate = aggregate
        if num_tokens > 0:
            if location in ["first", "middle", "last"]:
                self.tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
            elif location == "bilateral":
                assert num_tokens % 2 == 0
                self.tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
            elif location == "uniform":
                self.tokens = nn.Parameter(torch.zeros(1, num_tokens, dim))
            else:
                raise NotImplementedError
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
        if self.location == "first":
            tokens = self.tokens.expand(len(x), -1, -1)
            x = torch.concat([tokens, x], dim=1)
        elif self.location == "middle":
            tokens = self.tokens.expand(len(x), -1, -1)
            pre, post = x.chunk(chunks=2, dim=1)
            x = torch.concat([pre, tokens, post], dim=1)
        elif self.location == "last":
            tokens = self.tokens.expand(len(x), -1, -1)
            x = torch.concat([x, tokens], dim=1)
        elif self.location == "bilateral":
            first, last = self.tokens.chunk(chunks=2, dim=1)
            first = first.expand(len(x), -1, -1)
            last = last.expand(len(x), -1, -1)
            x = torch.concat([first, x, last], dim=1)
        elif self.location == "uniform":
            chunks = x.chunk(chunks=self.num_tokens + 1, dim=1)
            tokens = self.tokens.expand(len(x), -1, -1)
            # interweave chunk with token
            interweaved = [chunks[0]]
            for i in range(self.num_tokens):
                interweaved.append(tokens[:, i:i + 1])
                interweaved.append(chunks[i + 1])
            x = torch.concat(interweaved, dim=1)
        else:
            raise NotImplementedError
        return x

    @property
    def output_shape(self):
        if self.aggregate == "flatten":
            return self.dim * self.num_tokens,
        if self.aggregate == "mean":
            return self.dim,
        raise NotImplementedError

    def split(self, x):
        if self.num_tokens == 0:
            return None, x
        assert x.ndim == 3
        if self.location == "first":
            cls_tokens = x[:, :self.num_tokens]
            patch_tokens = x[:, self.num_tokens:]
        elif self.location == "middle":
            middle_start = (x.size(1) - self.num_tokens) // 2
            middle_end = middle_start + self.num_tokens
            cls_tokens = x[:, middle_start:middle_end]
            patch_tokens = torch.concat([x[:, :middle_start], x[:, middle_end:]], dim=1)
        elif self.location == "bilateral":
            cls_tokens = x[:, [0, -1]]
            patch_tokens = x[:, 1:-1]
        else:
            raise NotImplementedError
        return cls_tokens, patch_tokens

    def pool(self, x):
        if self.num_tokens == 0:
            raise NotImplementedError
        else:
            # extract tokens
            if self.location == "first":
                x = x[:, :self.num_tokens]
            elif self.location == "middle":
                middle = x.size(1) // 2
                half_num_tokens = self.num_tokens // 2
                start = middle - half_num_tokens
                end = start + self.num_tokens
                x = x[:, start:end]
            elif self.location == "last":
                x = x[:, -self.num_tokens:]
            elif self.location == "bilateral":
                num_tokens_half = self.num_tokens // 2
                x = torch.concat([x[:, :num_tokens_half], x[:, -num_tokens_half:]], dim=1)
            elif self.location == "uniform":
                # all but the last chunk are full
                chunk_size = (x.size(1) - self.num_tokens) // (self.num_tokens + 1) + 1
                x = torch.stack([x[:, (i + 1) * chunk_size + i] for i in range(self.num_tokens)], dim=1)
            else:
                raise NotImplementedError

            # aggregate if multiple tokens are used
            if self.aggregate == "flatten":
                return x.flatten(start_dim=1)
            elif self.aggregate == "mean":
                return x.mean(dim=1)
            else:
                raise NotImplementedError
