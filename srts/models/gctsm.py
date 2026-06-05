from __future__ import annotations

import torch
from torch import nn


class GCTSM(nn.Module):
    """Goal-conditioned trajectory semantic model."""

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        history_len: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_slots: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.history_len = history_len
        self.num_slots = num_slots
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.goal_proj = nn.Linear(state_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.slot_tokens = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, horizon + 1 + num_slots, hidden_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=max(1, num_layers // 2))
        self.dec_queries = nn.Parameter(torch.randn(1, horizon, hidden_dim) * 0.02)
        self.out = nn.Linear(hidden_dim, state_dim)

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.state_mean.copy_(mean.detach())
        self.state_std.copy_(std.detach().clamp_min(1e-6))

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.state_mean) / self.state_std

    def build_mask(self, batch: int, device: torch.device, history_mask_prob: float) -> torch.Tensor:
        mask = torch.zeros(batch, self.horizon, dtype=torch.bool, device=device)
        if history_mask_prob > 0:
            mask[:, : self.history_len] = torch.rand(batch, self.history_len, device=device) < history_mask_prob
        mask[:, self.history_len : self.horizon - 1] = True
        return mask

    def encode(
        self,
        states: torch.Tensor,
        history_mask_prob: float = 0.0,
        normalized: bool = False,
    ) -> torch.Tensor:
        if not normalized:
            states = self.normalize(states)
        b = states.shape[0]
        mask = self.build_mask(b, states.device, history_mask_prob)
        tokens = self.state_proj(states)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, self.horizon, -1), tokens)
        goal = self.goal_proj(states[:, -1]).unsqueeze(1)
        slots = self.slot_tokens.expand(b, -1, -1)
        x = torch.cat([tokens, goal, slots], dim=1) + self.pos[:, : self.horizon + 1 + self.num_slots]
        encoded = self.encoder(x)
        return encoded[:, -self.num_slots :]

    def forward(self, states: torch.Tensor, history_mask_prob: float = 0.3):
        normalized_states = self.normalize(states)
        slots = self.encode(normalized_states, history_mask_prob=history_mask_prob, normalized=True)
        queries = self.dec_queries.expand(states.shape[0], -1, -1)
        decoded = self.decoder(queries, slots)
        return self.out(decoded), slots, normalized_states
