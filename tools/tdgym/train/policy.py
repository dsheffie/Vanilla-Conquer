"""Actor-critic network for MacroEnv observations.

Built from matrix multiplies only: on Apple silicon PyTorch runs those through
Accelerate, which uses the SME matrix unit, while its CPU convolutions go to NEON
kernels. The map is cut into non-overlapping patches (a reshape, no copying), each
patch is embedded with a shared linear layer, and a small transformer mixes the patches.
"""

import torch
from torch import nn

PATCH = 8


class Policy(nn.Module):
    def __init__(self, planes, grid, features, actions, dim=128, layers=2, heads=4, hidden=256):
        super().__init__()
        self.tokens = (grid // PATCH) ** 2
        self.patch_embed = nn.Linear(planes * PATCH * PATCH, dim)
        self.position = nn.Parameter(torch.zeros(1, self.tokens, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, layers)
        self.map_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 32), nn.ReLU())
        self.features = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(self.tokens * 32 + hidden, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, actions)
        self.value = nn.Linear(hidden, 1)
        nn.init.normal_(self.position, std=0.02)
        nn.init.orthogonal_(self.policy.weight, 0.01)
        nn.init.zeros_(self.policy.bias)

    def patches(self, grid):
        """(batch, planes, H, W) -> (batch, tokens, planes * PATCH * PATCH)."""
        b, c, h, w = grid.shape
        x = grid.reshape(b, c, h // PATCH, PATCH, w // PATCH, PATCH)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // PATCH) * (w // PATCH), c * PATCH * PATCH)

    def forward(self, grid, features, mask):
        """Returns masked policy logits and value. grid is uint8 planes, mask 1 = allowed."""
        x = self.patch_embed(self.patches(grid.float() / 16.0)) + self.position
        x = self.map_out(self.mixer(x)).flatten(1)
        h = self.trunk(torch.cat([x, self.features(features)], dim=1))
        logits = self.policy(h).masked_fill(mask == 0, -1e8)
        return logits, self.value(h).squeeze(-1)
