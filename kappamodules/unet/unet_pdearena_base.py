import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation=nn.GELU) -> None:
        super().__init__()
        self.activation = activation()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if norm:
            # Original used BatchNorm2d
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.activation(self.norm2(self.conv2(h)))
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation=nn.GELU) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        h = self.pool(x)
        h = self.conv(h)
        return h


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation=nn.GELU) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.up(x1)
        h = torch.cat([x2, h], dim=1)
        h = self.conv(h)
        return h


class UnetPdearenaBase(nn.Module):
    """
    PDEArena implementation of the original U-Net architecture with input_dim/output_dim instead of the 6 properties
    https://github.com/microsoft/pdearena/blob/main/pdearena/modules/twod_unetbase.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.image_proj = ConvBlock(input_dim, hidden_dim)
        self.down = nn.ModuleList(
            [
                Down(hidden_dim, hidden_dim * 2),
                Down(hidden_dim * 2, hidden_dim * 4),
                Down(hidden_dim * 4, hidden_dim * 8),
                Down(hidden_dim * 8, hidden_dim * 16),
            ]
        )
        self.up = nn.ModuleList(
            [
                Up(hidden_dim * 16, hidden_dim * 8),
                Up(hidden_dim * 8, hidden_dim * 4),
                Up(hidden_dim * 4, hidden_dim * 2),
                Up(hidden_dim * 2, hidden_dim),
            ]
        )
        # should there be a final norm too? but we aren't doing "prenorm" in the original
        self.final = nn.Conv2d(hidden_dim, output_dim, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        h = self.image_proj(x)

        x1 = self.down[0](h)
        x2 = self.down[1](x1)
        x3 = self.down[2](x2)
        x4 = self.down[3](x3)
        
        x = self.up[0](x4, x3)
        x = self.up[1](x, x2)
        x = self.up[2](x, x1)
        x = self.up[3](x, h)

        x = self.final(x)
        return x
