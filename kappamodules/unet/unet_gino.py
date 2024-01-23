import torch.nn.functional as F
from torch import nn
import torch


class UnetGino(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=4, num_groups=8, output_dim=None):
        super().__init__()
        self.dim = hidden_dim
        self.depth = depth
        self.num_groups = num_groups
        self.output_dim = output_dim

        # create dim per level
        assert depth > 1
        assert isinstance(hidden_dim, int) and hidden_dim % 2 == 0
        dim_per_level = [hidden_dim * 2 ** i for i in range(depth)]

        # create down path
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                # first block has no pooling and goes from input_dim -> dim_per_level[0] -> dim_per_level[1]
                self.down_blocks.append(
                    nn.Sequential(
                        # pooling
                        nn.Identity(),
                        # conv1
                        nn.GroupNorm(num_groups=1, num_channels=input_dim),
                        nn.Conv3d(input_dim, dim_per_level[0] // 2, kernel_size=3, padding=1, bias=False),
                        nn.ReLU(),
                        # conv2
                        nn.GroupNorm(num_groups=num_groups, num_channels=dim_per_level[0] // 2),
                        nn.Conv3d(dim_per_level[0] // 2, dim_per_level[0], kernel_size=3, padding=1, bias=False),
                        nn.ReLU(),
                    ),
                )
            else:
                self.down_blocks.append(
                    nn.Sequential(
                        # pooling
                        nn.MaxPool3d(kernel_size=2, stride=2),
                        # conv1
                        nn.GroupNorm(num_groups=num_groups, num_channels=dim_per_level[i] // 2),
                        nn.Conv3d(dim_per_level[i] // 2, dim_per_level[i] // 2, kernel_size=3, padding=1, bias=False),
                        nn.ReLU(),
                        # conv2
                        nn.GroupNorm(num_groups=num_groups, num_channels=dim_per_level[i] // 2),
                        nn.Conv3d(dim_per_level[i] // 2, dim_per_level[i], kernel_size=3, padding=1, bias=False),
                        nn.ReLU(),
                    ),
                )

        # create up blocks
        self.up_blocks = nn.ModuleList()
        rev_dim_per_level = list(reversed(dim_per_level))
        for i in range(depth - 1):
            self.up_blocks.append(
                nn.Sequential(
                    # conv1
                    nn.GroupNorm(num_groups=num_groups, num_channels=rev_dim_per_level[i] + rev_dim_per_level[i + 1]),
                    nn.Conv3d(
                        rev_dim_per_level[i] + rev_dim_per_level[i + 1],
                        rev_dim_per_level[i + 1],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                    # conv2
                    nn.GroupNorm(num_groups=num_groups, num_channels=rev_dim_per_level[i + 1]),
                    nn.Conv3d(
                        rev_dim_per_level[i + 1],
                        rev_dim_per_level[i + 1],
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                ),
            )

        self.pred = nn.Conv3d(rev_dim_per_level[-1], output_dim or hidden_dim, kernel_size=1)

    def forward(self, x):
        stack = []

        # down path
        for down_block in self.down_blocks:
            x = down_block(x)
            stack.append(x)

        # x == stack[-1] -> pop last element
        stack.pop()

        # up path
        for up_block in self.up_blocks:
            residual = stack.pop()
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.concat([residual, x], dim=1)
            x = up_block(x)

        # pred
        x = self.pred(x)
        return x