from __future__ import annotations

import numpy as np
import torch

from srts.data.trajectories import OfflineDataset, make_segments
from srts.models.diffusion import ConditionalBridgeDenoiser, DDPMBridge
from srts.models.dynamics import ForwardDynamics, InverseDynamics, RewardModel
from srts.models.gctsm import GCTSM
from srts.stitching import build_bridge_dataset, compute_embeddings, select_pairs
from srts.train import train_bridge, train_dynamics, train_gctsm
from srts.utils.seed import seed_everything


def synthetic_dataset(n: int = 256, state_dim: int = 5, action_dim: int = 2) -> OfflineDataset:
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, state_dim)).astype(np.float32)
    actions = rng.normal(size=(n, action_dim)).astype(np.float32)
    drift = 0.05 * rng.normal(size=(n, state_dim)).astype(np.float32)
    next_obs = obs + drift
    rewards = -(actions**2).sum(axis=1, keepdims=True).astype(np.float32)
    terminals = np.zeros(n, dtype=bool)
    terminals[63::64] = True
    return OfflineDataset(obs, actions, rewards, next_obs, terminals)


def main() -> None:
    seed_everything(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = synthetic_dataset()
    H, P, L = 3, 4, 3
    segments = make_segments(dataset, H, P)

    gctsm = GCTSM(dataset.state_dim, H + P, H, hidden_dim=32, num_layers=1, num_heads=4, num_slots=2)
    states = torch.as_tensor(dataset.observations, dtype=torch.float32)
    gctsm.set_normalizer(states.mean(dim=0), states.std(dim=0))
    train_gctsm(gctsm, segments.states, {"batch_size": 16, "epochs": 1, "lr": 1e-3, "history_mask_prob": 0.2}, device)

    bridge = DDPMBridge(ConditionalBridgeDenoiser(dataset.state_dim, L + 2, hidden_dim=32), steps=4)
    bridge.set_normalizer(states.mean(dim=0), states.std(dim=0))
    normalized = (states - states.mean(dim=0)) / states.std(dim=0).clamp_min(1e-6)
    bridge.set_clip_bounds(torch.quantile(normalized, 0.01, dim=0), torch.quantile(normalized, 0.99, dim=0))
    bridge_segments = make_segments(dataset, 1, L + 1).states
    train_bridge(bridge, bridge_segments, {"batch_size": 16, "epochs": 1, "lr": 1e-3}, device)

    inv = InverseDynamics(dataset.state_dim, dataset.action_dim, hidden_dim=32)
    rew = RewardModel(dataset.state_dim, dataset.action_dim, hidden_dim=32)
    fwd = ForwardDynamics(dataset.state_dim, dataset.action_dim, hidden_dim=32)
    calibration = train_dynamics(
        inv,
        rew,
        fwd,
        dataset,
        {
            "batch_size": 32,
            "epochs": 1,
            "lr": 1e-3,
            "validation_fraction": 0.1,
            "threshold_percentile": 100,
        },
        device,
    )

    emb = compute_embeddings(gctsm, segments.states, device, batch_size=32)
    pairs = select_pairs(emb, segments.boundary, segments.trajectory_ids, top_k=8, num_queries=8)
    augmented, _ = build_bridge_dataset(
        dataset,
        segments,
        pairs,
        bridge,
        inv,
        rew,
        fwd,
        threshold=max(100.0, calibration["consistency_threshold"]),
        max_transitions=32,
        device=device,
    )
    assert len(augmented.observations) > len(dataset.observations)
    print("smoke test passed")


if __name__ == "__main__":
    main()
