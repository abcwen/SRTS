from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1)
    alpha_bars = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5).pow(2)
    alpha_bars = alpha_bars / alpha_bars[0]
    return (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(1e-5, 0.999)


class ConditionalResidualBlock1D(nn.Module):
    """Residual temporal convolution block with FiLM conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.silu(self.norm1(h))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.conv2(h)
        h = F.silu(self.norm2(h))
        return h + self.residual(x)


class ConditionalBridgeDenoiser(nn.Module):
    """Temporal 1D U-Net conditioned on timestep and two boundary states."""

    def __init__(self, state_dim: int, horizon: int, hidden_dim: int = 128, time_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.time_dim = time_dim
        cond_dim = hidden_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(time_dim + 2 * state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

        c1, c2, c3 = hidden_dim, hidden_dim * 2, hidden_dim * 4
        self.input_proj = nn.Conv1d(state_dim, c1, kernel_size=3, padding=1)
        self.down1 = ConditionalResidualBlock1D(c1, c1, cond_dim)
        self.downsample1 = nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.down2 = ConditionalResidualBlock1D(c2, c2, cond_dim)
        self.downsample2 = nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1)

        self.mid1 = ConditionalResidualBlock1D(c3, c3, cond_dim)
        self.mid2 = ConditionalResidualBlock1D(c3, c3, cond_dim)

        self.upsample2 = nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.up2 = ConditionalResidualBlock1D(c2 + c2, c2, cond_dim)
        self.upsample1 = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.up1 = ConditionalResidualBlock1D(c1 + c1, c1, cond_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(group_count(c1), c1),
            nn.SiLU(),
            nn.Conv1d(c1, state_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        original_horizon = x.shape[1]
        pad = (-original_horizon) % 4
        if pad:
            x = F.pad(x, (0, 0, 0, pad))

        cond = self.cond_encoder(torch.cat([timestep_embedding(t, self.time_dim), start, end], dim=-1))
        h0 = self.input_proj(x.transpose(1, 2))
        h1 = self.down1(h0, cond)
        h2 = self.down2(self.downsample1(h1), cond)
        h3 = self.downsample2(h2)

        h = self.mid2(self.mid1(h3, cond), cond)
        h = self.upsample2(h)
        h = F.interpolate(h, size=h2.shape[-1], mode="nearest")
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = F.interpolate(h, size=h1.shape[-1], mode="nearest")
        h = self.up1(torch.cat([h, h1], dim=1), cond)
        return self.output(h).transpose(1, 2)[:, :original_horizon]


class DDPMBridge(nn.Module):
    """Conditional trajectory DDPM with endpoint inpainting and EMA sampling."""

    def __init__(
        self,
        denoiser: ConditionalBridgeDenoiser,
        steps: int = 100,
        ema_decay: float = 0.995,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(denoiser).requires_grad_(False)
        self.ema_decay = ema_decay
        self.steps = steps
        betas = cosine_beta_schedule(steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef1 = betas * alpha_bars_prev.sqrt() / (1.0 - alpha_bars)
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * alphas.sqrt() / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("state_mean", torch.zeros(1, 1, denoiser.state_dim))
        self.register_buffer("state_std", torch.ones(1, 1, denoiser.state_dim))
        self.register_buffer("state_clip_low", torch.full((1, 1, denoiser.state_dim), -5.0))
        self.register_buffer("state_clip_high", torch.full((1, 1, denoiser.state_dim), 5.0))

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.state_mean.copy_(mean.detach().reshape(1, 1, -1))
        self.state_std.copy_(std.detach().reshape(1, 1, -1).clamp_min(1e-6))

    def set_clip_bounds(self, low: torch.Tensor, high: torch.Tensor) -> None:
        self.state_clip_low.copy_(low.detach().reshape(1, 1, -1))
        self.state_clip_high.copy_(high.detach().reshape(1, 1, -1))

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.state_mean) / self.state_std

    def denormalize(self, states: torch.Tensor) -> torch.Tensor:
        return states * self.state_std + self.state_mean

    @torch.no_grad()
    def update_ema(self) -> None:
        for ema_param, param in zip(self.ema_denoiser.parameters(), self.denoiser.parameters()):
            ema_param.data.lerp_(param.data, 1.0 - self.ema_decay)

    @staticmethod
    def apply_boundary_inpainting(x: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        x[:, 0] = start
        x[:, -1] = end
        return x

    def loss(self, clean: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        b = clean.shape[0]
        full_clean = torch.cat([start[:, None], clean, end[:, None]], dim=1)
        full_clean = self.normalize(full_clean)
        start_norm, end_norm = full_clean[:, 0], full_clean[:, -1]
        t = torch.randint(0, self.steps, (b,), device=clean.device)
        noise = torch.randn_like(full_clean)
        noisy = (
            self.sqrt_alpha_bars[t].view(b, 1, 1) * full_clean
            + self.sqrt_one_minus_alpha_bars[t].view(b, 1, 1) * noise
        )
        noisy = self.apply_boundary_inpainting(noisy, start_norm, end_norm)
        pred = self.denoiser(noisy, t, start_norm, end_norm)
        return F.mse_loss(pred[:, 1:-1], noise[:, 1:-1])

    @torch.no_grad()
    def sample(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        b = start.shape[0]
        start_norm = self.normalize(start[:, None])[:, 0]
        end_norm = self.normalize(end[:, None])[:, 0]
        x = torch.randn(b, self.denoiser.horizon, self.denoiser.state_dim, device=start.device)
        x = self.apply_boundary_inpainting(x, start_norm, end_norm)
        denoiser = self.ema_denoiser.eval()
        for idx in reversed(range(self.steps)):
            t = torch.full((b,), idx, device=start.device, dtype=torch.long)
            pred_noise = denoiser(x, t, start_norm, end_norm)
            pred_x0 = (
                x - self.sqrt_one_minus_alpha_bars[idx] * pred_noise
            ) / self.sqrt_alpha_bars[idx]
            pred_x0 = torch.maximum(torch.minimum(pred_x0, self.state_clip_high), self.state_clip_low)
            mean = self.posterior_mean_coef1[idx] * pred_x0 + self.posterior_mean_coef2[idx] * x
            if idx > 0:
                x = mean + self.posterior_variance[idx].sqrt() * torch.randn_like(x)
            else:
                x = mean
            x = self.apply_boundary_inpainting(x, start_norm, end_norm)
        return self.denormalize(x)
