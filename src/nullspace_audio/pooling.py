from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_mask(mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
    """Return float32 mask of shape [B, T, 1] matching hidden [B, T, D]."""
    if mask is None:
        return torch.ones(
            hidden.shape[0], hidden.shape[1], 1, dtype=torch.float32, device=hidden.device
        )
    if mask.ndim != 2 or mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"Expected mask of shape {tuple(hidden.shape[:2])}, got {tuple(mask.shape)}"
        )
    return mask.to(dtype=torch.float32).unsqueeze(-1)


def masked_mean_pool(
    hidden: torch.Tensor,
    mask: torch.Tensor | None = None,
    l2_normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute masked temporal mean pooling over [B, T, D] -> [B, D]."""
    x = hidden.to(dtype=torch.float32)
    m = _expand_mask(mask, x)
    counts = m.sum(dim=1).clamp(min=eps)
    mean = (x * m).sum(dim=1) / counts
    if l2_normalize:
        mean = F.normalize(mean, p=2, dim=-1, eps=eps)
    return mean


def masked_mean_std_pool(
    hidden: torch.Tensor,
    mask: torch.Tensor | None = None,
    l2_normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute masked temporal [mean; std] pooling over [B, T, D] -> [B, 2*D]."""
    x = hidden.to(dtype=torch.float32)
    m = _expand_mask(mask, x)
    counts = m.sum(dim=1).clamp(min=eps)
    mean = (x * m).sum(dim=1) / counts
    diff_sq = ((x - mean.unsqueeze(1)) ** 2) * m
    var = diff_sq.sum(dim=1) / counts
    std = torch.sqrt(var.clamp(min=eps))
    pooled = torch.cat([mean, std], dim=-1)
    if l2_normalize:
        pooled = F.normalize(pooled, p=2, dim=-1, eps=eps)
    return pooled
