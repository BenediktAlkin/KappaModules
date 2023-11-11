from torch import nn

from .vit_block import VitBlock


class VitBlocks(nn.Module):
    def __init__(self, depth, dim, num_heads, eps=1e-6):
        super().__init__()
        self.blocks = nn.Sequential(*[
            VitBlock(
                dim=dim,
                num_heads=num_heads,
                eps=eps,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x
