from torch import nn

from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from kappamodules.init import init_xavier_uniform_zero_bias


class TimestepEmbed(nn.Module):
    """ https://github.com/facebookresearch/DiT/blob/main/models.py#L27C1-L64C21 but more performant """

    def __init__(self, num_total_timesteps, embed_dim, hidden_dim=None):
        super().__init__()
        self.num_total_timesteps = num_total_timesteps
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim * 4
        # buffer/modules
        self.register_buffer("embed", get_sincos_1d_from_seqlen(seqlen=num_total_timesteps, dim=embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # init
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_xavier_uniform_zero_bias)

    def forward(self, timestep):
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten()
        embed = self.embed[timestep]
        return self.mlp(embed)
