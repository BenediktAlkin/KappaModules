import torch
from torch import nn

from .perceiver_block import PerceiverBlock


class PerceiverPoolingBlock(nn.Module):
    """
    implementation inspired by
    https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py#L74
    """

    def __init__(self, dim, num_heads, num_query_tokens, perceiver_kwargs=None, init_query="mean0std1"):
        super().__init__()
        self.init_query = init_query
        self.query = nn.Parameter(torch.empty(size=(num_query_tokens, dim)))
        self.perceiver = PerceiverBlock(dim=dim, num_heads=num_heads, **(perceiver_kwargs or {}))
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_query == "mean0std1":
            nn.init.trunc_normal_(self.query)
        else:
            raise NotImplementedError

    def forward(self, kv, attn_mask=None):
        query = self.query.expand(len(kv), -1, -1)
        return self.perceiver(q=query, kv=kv, attn_mask=attn_mask)
