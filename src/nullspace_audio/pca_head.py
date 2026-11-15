from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass
class PCABlock:
    """Frozen soft-whitened PCA state for one intermediate layer block."""

    mean: torch.Tensor        # [d_in]
    components: torch.Tensor  # [d_in, n_components]
    scale: torch.Tensor       # [n_components], sigma^(1/2) for soft whitening


def fit_balanced_pca(
    domain_features: Sequence[np.ndarray | torch.Tensor],
    n_components: int = 128,
    eps: float = 1e-5,
) -> PCABlock:
    """Fit a balanced, soft-whitened PCA across multiple external audio domains.

    Each domain in `domain_features` is weighted equally regardless of sample count.
    Soft whitening computes scale s = sigma^(1/2) where sigma is the component std
    (i.e. eigenvalue^(1/4)), so dividing by (s + eps) dampens dominant directions
    less aggressively than full whitening (sigma^-1).
    """
    if not domain_features:
        raise ValueError("domain_features must contain at least one domain array.")

    arrays: list[np.ndarray] = []
    for feat in domain_features:
        arr = feat.detach().cpu().numpy() if isinstance(feat, torch.Tensor) else np.asarray(feat)
        if arr.ndim != 2 or arr.shape[0] < 2:
            raise ValueError(f"Each domain array must be 2D with >= 2 samples, got {arr.shape}")
        arrays.append(arr.astype(np.float64, copy=False))

    d_in = arrays[0].shape[1]
    if any(a.shape[1] != d_in for a in arrays):
        raise ValueError("All domain feature arrays must have the same feature dimension.")

    n_domains = len(arrays)
    # Balanced global mean: average of per-domain means
    domain_means = [a.mean(axis=0) for a in arrays]
    global_mean = np.mean(domain_means, axis=0)

    # Balanced global covariance: average of per-domain second-moment matrices around global_mean
    cov = np.zeros((d_in, d_in), dtype=np.float64)
    for a in arrays:
        centered = a - global_mean[None, :]
        cov += (centered.T @ centered) / a.shape[0]
    cov /= n_domains

    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1][:n_components]
    top_vals = np.maximum(eigvals[idx], eps)
    top_vecs = eigvecs[:, idx]

    # Soft whitening: sigma = sqrt(eigval), scale s = sigma^(1/2)
    sigma = np.sqrt(top_vals)
    scale = np.sqrt(sigma)

    return PCABlock(
        mean=torch.from_numpy(np.ascontiguousarray(global_mean.astype(np.float32))),
        components=torch.from_numpy(np.ascontiguousarray(top_vecs.astype(np.float32))),
        scale=torch.from_numpy(np.ascontiguousarray(scale.astype(np.float32))),
    )


class NullSpaceAudioHead(nn.Module):
    """Frozen 256D null-space audio head for EmbeddingGemma-2."""

    def __init__(
        self,
        pca1: PCABlock,
        pca2: PCABlock,
        u_perp: torch.Tensor,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.eps = eps

        self.register_buffer("m1", pca1.mean.to(torch.float32))
        self.register_buffer("p1", pca1.components.to(torch.float32))
        self.register_buffer("s1", pca1.scale.to(torch.float32))

        self.register_buffer("m2", pca2.mean.to(torch.float32))
        self.register_buffer("p2", pca2.components.to(torch.float32))
        self.register_buffer("s2", pca2.scale.to(torch.float32))

        self.register_buffer("u_perp", u_perp.to(torch.float32))

    def project_block(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        proj: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project and soft-whiten one pooled layer block -> unit 128D vector."""
        z = ((x.to(torch.float32) - mean) @ proj) / (scale + self.eps)
        return F.normalize(z, p=2, dim=-1, eps=self.eps)

    def compute_shift(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute the 768D null-space shift vector (z @ U_perp^T)."""
        z1 = self.project_block(x1, self.m1, self.p1, self.s1)
        z2 = self.project_block(x2, self.m2, self.p2, self.s2)
        w = 0.5 ** 0.5
        z = F.normalize(torch.cat([w * z1, w * z2], dim=-1), p=2, dim=-1, eps=self.eps)
        return z @ self.u_perp.T

    def forward(
        self,
        v_base: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Inject the null-space acoustic shift into v_base and L2-normalize."""
        v_base_f32 = F.normalize(v_base.to(torch.float32), p=2, dim=-1, eps=self.eps)
        if gamma == 0.0:
            return v_base_f32
        shift = self.compute_shift(x1, x2)
        return F.normalize(v_base_f32 + gamma * shift, p=2, dim=-1, eps=self.eps)

    def save_safetensors(self, path: str | Path) -> None:
        """Save all frozen head tensors to a safetensors file."""
        state = {
            "m1": self.m1.detach().cpu().contiguous(),
            "p1": self.p1.detach().cpu().contiguous(),
            "s1": self.s1.detach().cpu().contiguous(),
            "m2": self.m2.detach().cpu().contiguous(),
            "p2": self.p2.detach().cpu().contiguous(),
            "s2": self.s2.detach().cpu().contiguous(),
            "u_perp": self.u_perp.detach().cpu().contiguous(),
        }
        save_file(state, str(path))

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        eps: float = 1e-5,
    ) -> NullSpaceAudioHead:
        """Load a frozen NullSpaceAudioHead from a safetensors file."""
        tensors = load_file(str(path))
        pca1 = PCABlock(mean=tensors["m1"], components=tensors["p1"], scale=tensors["s1"])
        pca2 = PCABlock(mean=tensors["m2"], components=tensors["p2"], scale=tensors["s2"])
        return cls(pca1=pca1, pca2=pca2, u_perp=tensors["u_perp"], eps=eps)
