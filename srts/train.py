from __future__ import annotations

from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from srts.data.trajectories import OfflineDataset, SegmentDataset, TransitionDataset
from srts.models.diffusion import DDPMBridge
from srts.models.dynamics import ForwardDynamics, InverseDynamics, RewardModel
from srts.models.gctsm import GCTSM


def split_dataset(dataset, cfg: dict):
    val_size = max(1, int(len(dataset) * cfg.get("validation_fraction", 0.1)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small for the requested validation split.")
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("split_seed", 42)),
    )


def improved(value: float, best: float, min_delta: float) -> bool:
    return value < best - min_delta


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_gctsm(
    model: GCTSM,
    segments,
    cfg: dict,
    device: torch.device,
    progress_path: Path | None = None,
    best_path: Path | None = None,
    resume: bool = False,
) -> None:
    train_data, val_data = split_dataset(SegmentDataset(segments), cfg)
    batch_size = cfg.get("batch_size", 256)
    epochs = cfg.get("epochs", 50)
    history_mask_prob = cfg.get("history_mask_prob", 0.3)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 3e-4))
    model.to(device).train()
    best_loss, stale = float("inf"), 0
    start_epoch = 0
    if resume and progress_path is not None and progress_path.exists():
        progress = torch.load(progress_path, map_location=device)
        model.load_state_dict(progress["model"])
        opt.load_state_dict(progress["optimizer"])
        best_loss, stale = progress["best_loss"], progress["stale"]
        if best_path is not None and not best_path.exists():
            best_loss, stale = float("inf"), 0
        start_epoch = progress["epoch"] + 1
        print(f"resuming GCTSM from epoch {start_epoch}")
    for epoch in range(start_epoch, epochs):
        losses = []
        for batch in tqdm(loader, desc=f"GCTSM {epoch + 1}/{epochs}", leave=False):
            batch = batch.to(device)
            recon, _, normalized = model(batch, history_mask_prob=history_mask_prob)
            loss = F.mse_loss(recon, normalized)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _, normalized = model(batch, history_mask_prob=0.0)
                val_losses.append(F.mse_loss(recon, normalized).item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"gctsm epoch={epoch + 1} loss={sum(losses) / len(losses):.6f} val={val_loss:.6f}")
        if improved(val_loss, best_loss, cfg.get("min_delta", 1e-5)):
            best_loss, stale = val_loss, 0
            if best_path is not None:
                atomic_torch_save(cpu_state_dict(model), best_path)
        else:
            stale += 1
        save_interval = cfg.get("progress_interval", 5)
        if progress_path is not None and ((epoch + 1) % save_interval == 0 or stale >= cfg.get("patience", 10)):
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_loss": best_loss,
                    "stale": stale,
                },
                progress_path,
            )
        if stale >= cfg.get("patience", 10):
            print(f"gctsm early stopping at epoch {epoch + 1}")
            break
        model.train()
    if best_path is not None and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))


def train_bridge(
    model: DDPMBridge,
    bridge_segments,
    cfg: dict,
    device: torch.device,
    progress_path: Path | None = None,
    best_path: Path | None = None,
    resume: bool = False,
) -> None:
    train_data, val_data = split_dataset(SegmentDataset(bridge_segments), cfg)
    batch_size = cfg.get("batch_size", 128)
    epochs = cfg.get("epochs", 80)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.denoiser.parameters(), lr=cfg.get("lr", 3e-4))
    model.to(device).train()
    best_loss, stale = float("inf"), 0
    start_epoch = 0
    if resume and progress_path is not None and progress_path.exists():
        progress = torch.load(progress_path, map_location=device)
        model.load_state_dict(progress["model"])
        opt.load_state_dict(progress["optimizer"])
        best_loss, stale = progress["best_loss"], progress["stale"]
        if best_path is not None and not best_path.exists():
            best_loss, stale = float("inf"), 0
        start_epoch = progress["epoch"] + 1
        print(f"resuming diffusion from epoch {start_epoch}")
    for epoch in range(start_epoch, epochs):
        losses = []
        for batch in tqdm(loader, desc=f"DDPM {epoch + 1}/{epochs}", leave=False):
            batch = batch.to(device)
            start, clean, end = batch[:, 0], batch[:, 1:-1], batch[:, -1]
            loss = model.loss(clean, start, end)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
            model.update_ema()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_losses.append(model.loss(batch[:, 1:-1], batch[:, 0], batch[:, -1]).item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"diffusion epoch={epoch + 1} loss={sum(losses) / len(losses):.6f} val={val_loss:.6f}")
        if improved(val_loss, best_loss, cfg.get("min_delta", 1e-5)):
            best_loss, stale = val_loss, 0
            if best_path is not None:
                atomic_torch_save(cpu_state_dict(model), best_path)
        else:
            stale += 1
        save_interval = cfg.get("progress_interval", 5)
        if progress_path is not None and ((epoch + 1) % save_interval == 0 or stale >= cfg.get("patience", 10)):
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_loss": best_loss,
                    "stale": stale,
                },
                progress_path,
            )
        if stale >= cfg.get("patience", 10):
            print(f"diffusion early stopping at epoch {epoch + 1}")
            break
        model.train()
    if best_path is not None and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))


def train_dynamics(
    inverse: InverseDynamics,
    reward: RewardModel,
    forward: ForwardDynamics,
    dataset: OfflineDataset,
    cfg: dict,
    device: torch.device,
    progress_path: Path | None = None,
    best_paths: dict[str, Path] | None = None,
    resume: bool = False,
) -> dict[str, float]:
    # Reward learning can use true terminal transitions, while inverse/forward
    # losses mask them because their next observation may be the next reset state.
    dynamics_data = TransitionDataset(dataset)
    print(
        f"dynamics data: using={len(dynamics_data)} "
        f"masked_next_state_terminals={int(dataset.terminals.sum())}"
    )
    train_data, val_data = split_dataset(dynamics_data, cfg)
    batch_size = cfg.get("batch_size", 512)
    epochs = cfg.get("epochs", 50)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    state_tensor = torch.as_tensor(dataset.observations, dtype=torch.float32)
    state_mean, state_std = state_tensor.mean(dim=0), state_tensor.std(dim=0).clamp_min(1e-6)
    inverse.set_normalizer(state_mean, state_std)
    reward.set_normalizer(state_mean, state_std)
    forward.set_normalizer(state_mean, state_std)
    params = list(inverse.parameters()) + list(reward.parameters()) + list(forward.parameters())
    opt = torch.optim.Adam(params, lr=cfg.get("lr", 3e-4))
    inverse.to(device).train()
    reward.to(device).train()
    forward.to(device).train()
    best_loss, stale = float("inf"), 0
    start_epoch = 0
    if resume and progress_path is not None and progress_path.exists():
        progress = torch.load(progress_path, map_location=device)
        inverse.load_state_dict(progress["inverse"])
        reward.load_state_dict(progress["reward"])
        forward.load_state_dict(progress["forward"])
        opt.load_state_dict(progress["optimizer"])
        best_loss, stale = progress["best_loss"], progress["stale"]
        if best_paths is not None and not all(path.exists() for path in best_paths.values()):
            best_loss, stale = float("inf"), 0
        start_epoch = progress["epoch"] + 1
        print(f"resuming dynamics from epoch {start_epoch}")
    for epoch in range(start_epoch, epochs):
        losses = []
        for s, a, r, ns, terminal in tqdm(loader, desc=f"Dynamics {epoch + 1}/{epochs}", leave=False):
            s, a, r, ns, terminal = s.to(device), a.to(device), r.to(device), ns.to(device), terminal.to(device)
            valid_next = ~terminal
            rew_loss = F.mse_loss(reward(s, a), r)
            if valid_next.any():
                inv_loss = F.mse_loss(inverse(s[valid_next], ns[valid_next]), a[valid_next])
                normalized_error = (forward(s[valid_next], a[valid_next]) - ns[valid_next]) / forward.state_std
                fwd_loss = F.mse_loss(normalized_error, torch.zeros_like(normalized_error))
            else:
                inv_loss = torch.zeros((), device=device)
                fwd_loss = torch.zeros((), device=device)
            loss = inv_loss + rew_loss + fwd_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        inverse.eval()
        reward.eval()
        forward.eval()
        val_losses = []
        with torch.no_grad():
            for s, a, r, ns, terminal in val_loader:
                s, a, r, ns, terminal = s.to(device), a.to(device), r.to(device), ns.to(device), terminal.to(device)
                valid_next = ~terminal
                batch_loss = F.mse_loss(reward(s, a), r)
                if valid_next.any():
                    normalized_error = (forward(s[valid_next], a[valid_next]) - ns[valid_next]) / forward.state_std
                    batch_loss = (
                        batch_loss
                        + F.mse_loss(inverse(s[valid_next], ns[valid_next]), a[valid_next])
                        + F.mse_loss(normalized_error, torch.zeros_like(normalized_error))
                    )
                val_losses.append(batch_loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"dynamics epoch={epoch + 1} loss={sum(losses) / len(losses):.6f} val={val_loss:.6f}")
        if improved(val_loss, best_loss, cfg.get("min_delta", 1e-5)):
            best_loss = val_loss
            stale = 0
            if best_paths is not None:
                atomic_torch_save(cpu_state_dict(inverse), best_paths["inverse"])
                atomic_torch_save(cpu_state_dict(reward), best_paths["reward"])
                atomic_torch_save(cpu_state_dict(forward), best_paths["forward"])
        else:
            stale += 1
        save_interval = cfg.get("progress_interval", 5)
        if progress_path is not None and ((epoch + 1) % save_interval == 0 or stale >= cfg.get("patience", 10)):
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "inverse": inverse.state_dict(),
                    "reward": reward.state_dict(),
                    "forward": forward.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_loss": best_loss,
                    "stale": stale,
                },
                progress_path,
            )
        if stale >= cfg.get("patience", 10):
            print(f"dynamics early stopping at epoch {epoch + 1}")
            break
        inverse.train()
        reward.train()
        forward.train()

    if best_paths is not None:
        inverse.load_state_dict(torch.load(best_paths["inverse"], map_location=device))
        reward.load_state_dict(torch.load(best_paths["reward"], map_location=device))
        forward.load_state_dict(torch.load(best_paths["forward"], map_location=device))

    forward.eval()
    errors = []
    with torch.no_grad():
        for s, a, _, ns, terminal in val_loader:
            s, a, ns, terminal = s.to(device), a.to(device), ns.to(device), terminal.to(device)
            valid_next = ~terminal
            if valid_next.any():
                errors.append(forward.normalized_error(s[valid_next], a[valid_next], ns[valid_next]).cpu())
    if not errors:
        raise RuntimeError("Dynamics validation split contains no non-terminal transitions.")
    errors = torch.cat(errors)
    error_q = cfg.get("threshold_percentile", 90.0) / 100.0
    metrics = {
        "consistency_threshold": float(torch.quantile(errors, error_q).item()),
        "validation_error_mean": float(errors.mean().item()),
    }
    print(
        "dynamics calibration: "
        f"error_mean={metrics['validation_error_mean']:.4f} "
        f"error_threshold={metrics['consistency_threshold']:.4f}"
    )
    return metrics
