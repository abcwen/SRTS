from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from srts.data.trajectories import OfflineDataset, SegmentBatch, SegmentDataset
from srts.models.diffusion import DDPMBridge
from srts.models.dynamics import ForwardDynamics, InverseDynamics, RewardModel
from srts.models.gctsm import GCTSM


@torch.no_grad()
def compute_embeddings(model: GCTSM, segments: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval().to(device)
    vectors = []
    for batch in DataLoader(SegmentDataset(segments), batch_size=batch_size):
        slots = model.encode(batch.to(device), history_mask_prob=0.0)
        vectors.append(slots.mean(dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def select_pairs(
    embeddings: np.ndarray,
    boundaries: np.ndarray,
    trajectory_ids: np.ndarray,
    top_k: int,
    num_queries: int,
    use_faiss: bool = False,
    min_semantic_similarity: float | None = None,
    max_boundary_distance: float | None = None,
) -> list[tuple[int, int]]:
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError("At least two trajectory segments are required for stitching.")
    k = min(top_k, n - 1)
    query_ids = np.random.choice(n, size=min(num_queries, n), replace=False)
    z = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    b_mean, b_std = boundaries.mean(axis=0), boundaries.std(axis=0) + 1e-6
    norm_boundaries = (boundaries - b_mean) / b_std
    pairs = []
    rejected_semantic = 0
    rejected_boundary = 0
    faiss_candidates = None
    if use_faiss:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FAISS retrieval requires `pip install faiss-cpu` or `faiss-gpu`.") from exc
        index = faiss.IndexFlatIP(z.shape[1])
        index.add(z.astype(np.float32))
        search_k = min(n, max(k * 4, k + 1))
        _, faiss_candidates = index.search(z[query_ids].astype(np.float32), search_k)
    for query_position, i in enumerate(query_ids):
        if use_faiss:
            retrieved = faiss_candidates[query_position]
            cand = retrieved[trajectory_ids[retrieved] != trajectory_ids[i]][:k]
            if cand.size < min(k, n - np.sum(trajectory_ids == trajectory_ids[i])):
                sim = z @ z[i]
                sim[trajectory_ids == trajectory_ids[i]] = -np.inf
                valid = np.flatnonzero(np.isfinite(sim))
                candidate_k = min(k, valid.size)
                valid_sim = sim[valid]
                cand = valid[np.argpartition(-valid_sim, kth=candidate_k - 1)[:candidate_k]]
        else:
            sim = z @ z[i]
            sim[trajectory_ids == trajectory_ids[i]] = -np.inf
            valid = np.flatnonzero(np.isfinite(sim))
            if valid.size == 0:
                continue
            candidate_k = min(k, valid.size)
            valid_sim = sim[valid]
            cand = valid[np.argpartition(-valid_sim, kth=candidate_k - 1)[:candidate_k]]
        candidate_similarity = z[cand] @ z[i]
        if min_semantic_similarity is not None:
            semantic_mask = candidate_similarity >= min_semantic_similarity
            cand = cand[semantic_mask]
            candidate_similarity = candidate_similarity[semantic_mask]
        if cand.size == 0:
            rejected_semantic += 1
            continue

        candidate_distance = np.linalg.norm(norm_boundaries[cand] - norm_boundaries[i], axis=1)
        if max_boundary_distance is not None:
            boundary_mask = candidate_distance <= max_boundary_distance
            cand = cand[boundary_mask]
            candidate_distance = candidate_distance[boundary_mask]
        if cand.size == 0:
            rejected_boundary += 1
            continue

        selected = int(cand[int(np.argmin(candidate_distance))])
        pairs.append((int(i), selected))
    if not pairs:
        raise RuntimeError(
            "No cross-trajectory candidate pairs passed selection. "
            f"rejected_semantic={rejected_semantic}, rejected_boundary={rejected_boundary}. "
            "Relax the pair quality thresholds."
        )
    print(
        f"pair selection: accepted={len(pairs)} "
        f"rejected_semantic={rejected_semantic} "
        f"rejected_boundary={rejected_boundary}"
    )
    return pairs


@torch.no_grad()
def build_bridge_dataset(
    original: OfflineDataset,
    segments: SegmentBatch,
    pairs: list[tuple[int, int]],
    bridge: DDPMBridge,
    inverse: InverseDynamics,
    reward: RewardModel,
    forward: ForwardDynamics,
    threshold: float,
    max_transitions: int,
    device: torch.device,
) -> tuple[OfflineDataset, dict[str, float]]:
    bridge.eval().to(device)
    inverse.eval().to(device)
    reward.eval().to(device)
    forward.eval().to(device)

    obs, acts, rews, next_obs = [], [], [], []
    rejected_errors = []
    rejected_actions = 0
    action_min = torch.as_tensor(original.action_min, dtype=torch.float32, device=device)
    action_max = torch.as_tensor(original.action_max, dtype=torch.float32, device=device)
    for i, j in pairs:
        start = torch.as_tensor(segments.boundary[i : i + 1], dtype=torch.float32, device=device)
        end = torch.as_tensor(segments.boundary[j : j + 1], dtype=torch.float32, device=device)
        states = bridge.sample(start, end)[0]
        for t in range(states.shape[0] - 1):
            s = states[t : t + 1]
            ns = states[t + 1 : t + 2]
            a = inverse(s, ns)
            if torch.any(a < action_min) or torch.any(a > action_max):
                rejected_actions += 1
                continue
            a = a.clamp(action_min, action_max)
            err = forward.normalized_error(s, a, ns)
            if err.item() < threshold:
                r = reward(s, a)
                obs.append(s.cpu().numpy()[0])
                acts.append(a.cpu().numpy()[0])
                rews.append(r.cpu().numpy()[0])
                next_obs.append(ns.cpu().numpy()[0])
            else:
                rejected_errors.append(float(err.item()))
        if len(obs) >= max_transitions:
            break

    if not obs:
        if rejected_errors:
            errors = np.asarray(rejected_errors, dtype=np.float32)
            raise RuntimeError(
                "No bridge transitions survived filtering. "
                f"threshold={threshold}, "
                f"forward_error_min={errors.min():.4f}, "
                f"forward_error_mean={errors.mean():.4f}, "
                f"forward_error_p50={np.percentile(errors, 50):.4f}, "
                f"forward_error_p90={np.percentile(errors, 90):.4f}. "
                "Increase stitching.consistency_threshold near the lower error quantiles, "
                "or train diffusion/dynamics longer."
            )
        raise RuntimeError(
            "No bridge transitions survived filtering. "
            f"rejected_action={rejected_actions}. "
            "Check action support, selected pairs, and bridge generation."
        )
    total_checked = len(obs) + len(rejected_errors) + rejected_actions
    filter_metrics = {
        "bridge_accepted": len(obs),
        "bridge_rejected_error": len(rejected_errors),
        "bridge_rejected_action": rejected_actions,
        "bridge_acceptance_rate": len(obs) / max(total_checked, 1),
    }
    print(
        f"bridge filtering: accepted={len(obs)} "
        f"rejected_error={len(rejected_errors)} "
        f"rejected_action={rejected_actions} "
        f"acceptance_rate={filter_metrics['bridge_acceptance_rate']:.4f}"
    )
    bridge_data = OfflineDataset(
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(acts, dtype=np.float32),
        rewards=np.asarray(rews, dtype=np.float32).reshape(-1, 1),
        next_observations=np.asarray(next_obs, dtype=np.float32),
        terminals=np.zeros(len(obs), dtype=bool),
        episode_ends=np.zeros(len(obs), dtype=bool),
    )
    return concat_datasets(original, bridge_data), filter_metrics


def concat_datasets(a: OfflineDataset, b: OfflineDataset) -> OfflineDataset:
    return OfflineDataset(
        observations=np.concatenate([a.observations, b.observations], axis=0),
        actions=np.concatenate([a.actions, b.actions], axis=0),
        rewards=np.concatenate([a.rewards, b.rewards], axis=0),
        next_observations=np.concatenate([a.next_observations, b.next_observations], axis=0),
        terminals=np.concatenate([a.terminals, b.terminals], axis=0),
        episode_ends=np.concatenate([a.episode_ends, b.episode_ends], axis=0),
    )
