from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from srts.data.d4rl_loader import load_d4rl_dataset
from srts.data.trajectories import load_npz_dataset, make_segments, save_npz_dataset
from srts.models.diffusion import ConditionalBridgeDenoiser, DDPMBridge
from srts.models.dynamics import ForwardDynamics, InverseDynamics, RewardModel
from srts.models.gctsm import GCTSM
from srts.stitching import build_bridge_dataset, compute_embeddings, select_pairs
from srts.train import train_bridge, train_dynamics, train_gctsm
from srts.utils.config import load_config, merge_overrides
from srts.utils.seed import seed_everything

IMPLEMENTATION_VERSION = "srts-final-v2"


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))


def make_output_path(run_dir: Path, configured_output: str, dataset_source: str, seed: int) -> Path:
    configured = Path(configured_output)
    suffix = configured.suffix or ".npz"
    stem = configured.stem or "augmented_srts"
    if stem == "augmented_srts":
        stem = "augmented"
    filename = (
        f"{safe_name(dataset_source)}_"
        f"{seed}_"
        f"{safe_name(stem)}_"
        f"{suffix}"
    )
    return run_dir / filename


def dataset_signature(dataset, source: str) -> dict:
    digest = hashlib.sha256()
    for array in (
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.next_observations,
        dataset.terminals,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
    return {
        "source": source,
        "transitions": len(dataset.observations),
        "content_sha256": digest.hexdigest(),
    }


def code_signature() -> str:
    root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "scripts/run_srts.py",
        "srts/train.py",
        "srts/stitching.py",
        "srts/data/trajectories.py",
        "srts/models/gctsm.py",
        "srts/models/diffusion.py",
        "srts/models/dynamics.py",
    ]
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_or_validate_manifest(path: Path, payload: dict, require_match: bool) -> str:
    fingerprint = make_fingerprint(payload)
    manifest = {"fingerprint": fingerprint, **payload}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Experiment fingerprint mismatch for {path.parent}. "
                "Refusing to load incompatible checkpoints."
            )
    elif require_match:
        raise FileNotFoundError(f"Resume/skip requested but experiment manifest is missing: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return fingerprint


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"loaded checkpoint: {path}")


def should_load_stage(skip: bool, resume: bool, paths: list[Path]) -> bool:
    available = all(path.exists() for path in paths)
    if skip and not available:
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"Requested skipped stage has missing checkpoints: {missing}")
    return skip or (resume and available)


def remove_completed_progress(path: Path) -> None:
    if path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce SRTS dataset augmentation.")
    parser.add_argument("--config", default="configs/configs.yaml")
    parser.add_argument("overrides", nargs="*", help="Override sensitivity keys, e.g. data.H=8 stitching.top_k=60")
    parser.add_argument("--use-faiss", action="store_true", help="Use FAISS for semantic Top-K retrieval.")
    parser.add_argument("--min-semantic-similarity", type=float, default=None)
    parser.add_argument("--max-boundary-distance", type=float, default=None)
    parser.add_argument("--run-root", default="runs", help="Root directory for isolated experiment outputs.")
    parser.add_argument("--resume", action="store_true", help="Reuse every completed stage checkpoint.")
    parser.add_argument("--skip-gctsm", action="store_true", help="Load GCTSM checkpoint without training.")
    parser.add_argument("--skip-diffusion", action="store_true", help="Load diffusion checkpoint without training.")
    parser.add_argument("--skip-dynamics", action="store_true", help="Load dynamics checkpoints without training.")
    args = parser.parse_args()
    cfg = merge_overrides(load_config(args.config), args.overrides)
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    dataset = load_npz_dataset(data_cfg["npz_path"]) if data_cfg.get("npz_path") else load_d4rl_dataset(data_cfg["dataset"])
    segment_stride = data_cfg.get("segment_stride", 1)
    segments = make_segments(dataset, data_cfg["H"], data_cfg["P"], segment_stride)
    dataset_source = Path(data_cfg["npz_path"]).stem if data_cfg.get("npz_path") else data_cfg["dataset"]
    fingerprint_payload = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "code_sha256": code_signature(),
        "dataset": dataset_signature(dataset, dataset_source),
        "state_dim": dataset.state_dim,
        "action_dim": dataset.action_dim,
        "config": cfg,
        "runtime_options": {
            "use_faiss": args.use_faiss,
            "min_semantic_similarity": args.min_semantic_similarity,
            "max_boundary_distance": args.max_boundary_distance,
        },
    }
    fingerprint = make_fingerprint(fingerprint_payload)
    run_dir = Path(args.run_root) / safe_name(dataset_source) / f"run_{fingerprint[:12]}"
    manifest_path = run_dir / "manifest.json"
    write_or_validate_manifest(
        manifest_path,
        fingerprint_payload,
        require_match=args.resume or args.skip_gctsm or args.skip_diffusion or args.skip_dynamics,
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    gctsm_path = checkpoint_dir / "best_gctsm.pt"
    bridge_path = checkpoint_dir / "best_bridge.pt"
    inverse_path = checkpoint_dir / "best_inverse_dynamics.pt"
    reward_path = checkpoint_dir / "best_reward_model.pt"
    forward_path = checkpoint_dir / "best_forward_dynamics.pt"
    calibration_path = checkpoint_dir / "dynamics_calibration.json"
    gctsm_progress_path = checkpoint_dir / "gctsm_progress.pt"
    bridge_progress_path = checkpoint_dir / "bridge_progress.pt"
    dynamics_progress_path = checkpoint_dir / "dynamics_progress.pt"

    state_dim, action_dim = dataset.state_dim, dataset.action_dim
    state_samples = torch.as_tensor(dataset.observations, dtype=torch.float32)
    state_mean = state_samples.mean(dim=0)
    state_std = state_samples.std(dim=0).clamp_min(1e-6)
    gctsm = GCTSM(
        state_dim=state_dim,
        horizon=data_cfg["H"] + data_cfg["P"],
        history_len=data_cfg["H"],
    )
    gctsm.set_normalizer(state_mean, state_std)
    if should_load_stage(args.skip_gctsm, args.resume, [gctsm_path]):
        load_checkpoint(gctsm, gctsm_path, device)
    else:
        train_gctsm(
            gctsm,
            segments.states,
            cfg.get("gctsm", {}),
            device,
            progress_path=gctsm_progress_path,
            best_path=gctsm_path,
            resume=args.resume,
        )
        remove_completed_progress(gctsm_progress_path)

    diffusion_cfg = cfg.get("diffusion", {})
    denoiser = ConditionalBridgeDenoiser(state_dim, data_cfg["bridge_horizon"] + 2)
    bridge = DDPMBridge(denoiser)
    bridge.set_normalizer(state_mean, state_std)
    normalized_states = (state_samples - state_mean) / state_std
    bridge.set_clip_bounds(
        torch.quantile(normalized_states, diffusion_cfg.get("clip_low_quantile", 0.005), dim=0),
        torch.quantile(normalized_states, diffusion_cfg.get("clip_high_quantile", 0.995), dim=0),
    )
    bridge_train_segments = make_segments(dataset, 1, data_cfg["bridge_horizon"] + 1, segment_stride).states
    if should_load_stage(args.skip_diffusion, args.resume, [bridge_path]):
        load_checkpoint(bridge, bridge_path, device)
    else:
        train_bridge(
            bridge,
            bridge_train_segments,
            diffusion_cfg,
            device,
            progress_path=bridge_progress_path,
            best_path=bridge_path,
            resume=args.resume,
        )
        remove_completed_progress(bridge_progress_path)

    inverse = InverseDynamics(state_dim, action_dim)
    reward = RewardModel(state_dim, action_dim)
    forward = ForwardDynamics(state_dim, action_dim)
    dynamics_paths = [inverse_path, reward_path, forward_path]
    if should_load_stage(args.skip_dynamics, args.resume, dynamics_paths):
        load_checkpoint(inverse, inverse_path, device)
        load_checkpoint(reward, reward_path, device)
        load_checkpoint(forward, forward_path, device)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    else:
        calibration = train_dynamics(
            inverse,
            reward,
            forward,
            dataset,
            cfg.get("dynamics", {}),
            device,
            progress_path=dynamics_progress_path,
            best_paths={"inverse": inverse_path, "reward": reward_path, "forward": forward_path},
            resume=args.resume,
        )
        remove_completed_progress(dynamics_progress_path)
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    embeddings = compute_embeddings(gctsm, segments.states, device)
    pairs = select_pairs(
        embeddings,
        segments.boundary,
        segments.trajectory_ids,
        top_k=cfg["stitching"]["top_k"],
        num_queries=cfg["stitching"].get("num_queries", 5000),
        use_faiss=args.use_faiss,
        min_semantic_similarity=args.min_semantic_similarity,
        max_boundary_distance=args.max_boundary_distance,
    )
    threshold_cfg = cfg["stitching"]["consistency_threshold"]
    if threshold_cfg == "auto" and "consistency_threshold" not in calibration:
        raise RuntimeError(
            "Automatic consistency threshold requires dynamics calibration. "
            "Run dynamics training once or set a fixed stitching.consistency_threshold."
        )
    consistency_threshold = (
        calibration["consistency_threshold"] if threshold_cfg == "auto" else float(threshold_cfg)
    )
    augmented, filter_metrics = build_bridge_dataset(
        dataset,
        segments,
        pairs,
        bridge,
        inverse,
        reward,
        forward,
        consistency_threshold,
        cfg["stitching"].get("max_bridge_transitions", 50000),
        device,
    )

    out = make_output_path(
        run_dir,
        data_cfg.get("output_path", "augmented_srts.npz"),
        dataset_source,
        cfg["seed"],
    )
    save_npz_dataset(str(out), augmented)
    report = {
        "dataset": data_cfg["dataset"],
        "seed": cfg["seed"],
        "original_transitions": len(dataset.observations),
        "augmented_transitions": len(augmented.observations),
        "cross_trajectory_pairs": len(pairs),
        "consistency_threshold": consistency_threshold,
        "use_faiss": args.use_faiss,
        "min_semantic_similarity": args.min_semantic_similarity,
        "max_boundary_distance": args.max_boundary_distance,
        "run_dir": str(run_dir),
        "fingerprint": fingerprint,
        "config": cfg,
        **calibration,
        **filter_metrics,
    }
    report_path = run_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved augmented dataset to {out.resolve()}")
    print(f"saved metrics to {report_path.resolve()}")
    print(f"original transitions={len(dataset.observations)} augmented transitions={len(augmented.observations)}")


if __name__ == "__main__":
    main()
