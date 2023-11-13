from torch import nn

from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen


class TimestepEmbed(nn.Module):
    """ https://github.com/facebookresearch/DiT/blob/main/models.py#L27C1-L64C21 but more performant """

    def __init__(self, num_total_timesteps, embed_dim, hidden_dim):
        super().__init__()
        self.num_total_timesteps = num_total_timesteps
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # buffer/modules
        self.register_buffer("embed", get_sincos_1d_from_seqlen(seqlen=num_total_timesteps, dim=embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, timestep):
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten()
        embed = self.embed[timestep]
        return self.mlp(embed)
