from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateNormalizedModel(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.state_mean.copy_(mean.detach())
        self.state_std.copy_(std.detach().clamp_min(1e-6))

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std


class InverseDynamics(StateNormalizedModel):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__(state_dim)
        self.model = MLP(state_dim * 2, action_dim, hidden_dim)

    def forward(self, s: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([self.normalize_state(s), self.normalize_state(ns)], dim=-1))


class RewardModel(StateNormalizedModel):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__(state_dim)
        self.model = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([self.normalize_state(s), a], dim=-1))


class ForwardDynamics(StateNormalizedModel):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__(state_dim)
        self.model = MLP(state_dim + action_dim, state_dim, hidden_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        normalized_delta = self.model(torch.cat([self.normalize_state(s), a], dim=-1))
        return s + normalized_delta * self.state_std

    def normalized_error(self, s: torch.Tensor, a: torch.Tensor, ns: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm((self(s, a) - ns) / self.state_std, dim=-1)
