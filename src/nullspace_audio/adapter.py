from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class AttentiveStatsPool(nn.Module):
    """Multi-head attentive statistics pooling over [B, T, D] -> [B, 2*D]."""

    def __init__(self, dim: int = 1024, num_heads: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, num_heads),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        x = hidden.to(torch.float32)  # [B, T, D]
        b, t, _ = x.shape
        logits = self.attn(x)  # [B, T, H]
        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=torch.bool)
            logits = logits.masked_fill(~m, float("-inf"))
        weights = torch.softmax(logits, dim=1)  # [B, T, H]

        # Reshape x into heads: [B, T, H, D_h]
        xh = x.view(b, t, self.num_heads, self.head_dim)
        w = weights.unsqueeze(-1)  # [B, T, H, 1]
        mean = (xh * w).sum(dim=1)  # [B, H, D_h]
        var = (((xh - mean.unsqueeze(1)) ** 2) * w).sum(dim=1)
        std = torch.sqrt(var.clamp(min=eps))

        pooled = torch.cat([mean.reshape(b, self.dim), std.reshape(b, self.dim)], dim=-1)
        return F.normalize(pooled, p=2, dim=-1, eps=eps)


class _BlockProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.to(torch.float32)), p=2, dim=-1)


class NullSpaceMLPAdapter(nn.Module):
    """Trainable Phase 1 MLP adapter constrained to the frozen 256D null space U_perp."""

    def __init__(
        self,
        u_perp: torch.Tensor,
        d_x1: int = 2048,
        d_x2: int = 1024,
        hidden_dim: int = 512,
        gamma_max: float = 1.5,
        dropout: float = 0.1,
        use_input_gate: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if u_perp.shape != (768, 256):
            raise ValueError(f"Expected u_perp shape (768, 256), got {tuple(u_perp.shape)}")
        self.gamma_max = gamma_max
        self.use_input_gate = use_input_gate
        self.eps = eps

        self.register_buffer("u_perp", u_perp.to(torch.float32).contiguous())
        self.proj1 = _BlockProjector(d_x1, hidden_dim, out_dim=128, dropout=dropout)
        self.proj2 = _BlockProjector(d_x2, hidden_dim, out_dim=128, dropout=dropout)
        self.block_logits = nn.Parameter(torch.zeros(2))

        if use_input_gate:
            self.gate = nn.Sequential(
                nn.Linear(d_x1 + d_x2, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
        else:
            self.gate_scalar = nn.Parameter(torch.tensor(0.0))

    def compute_gamma(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.use_input_gate:
            feats = torch.cat([x1.to(torch.float32), x2.to(torch.float32)], dim=-1)
            return self.gamma_max * torch.sigmoid(self.gate(feats))  # [B, 1]
        return self.gamma_max * torch.sigmoid(self.gate_scalar).view(1, 1)

    def compute_shift(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = self.proj1(x1)
        z2 = self.proj2(x2)
        weights = torch.softmax(self.block_logits, dim=0)
        z = F.normalize(
            torch.cat([torch.sqrt(weights[0]) * z1, torch.sqrt(weights[1]) * z2], dim=-1),
            p=2,
            dim=-1,
            eps=self.eps,
        )
        gamma = self.compute_gamma(x1, x2)
        shift = gamma * (z @ self.u_perp.T)
        return shift, gamma

    def forward(
        self,
        v_base: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        v_base_f32 = F.normalize(v_base.to(torch.float32), p=2, dim=-1, eps=self.eps)
        shift, _ = self.compute_shift(x1, x2)
        return F.normalize(v_base_f32 + shift, p=2, dim=-1, eps=self.eps)

    def save_safetensors(self, path: str | Path) -> None:
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(path))

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        d_x1: int = 2048,
        d_x2: int = 1024,
        hidden_dim: int = 512,
        gamma_max: float = 1.5,
        use_input_gate: bool = True,
    ) -> NullSpaceMLPAdapter:
        state = load_file(str(path))
        adapter = cls(
            u_perp=state["u_perp"],
            d_x1=d_x1,
            d_x2=d_x2,
            hidden_dim=hidden_dim,
            gamma_max=gamma_max,
            use_input_gate=use_input_gate,
        )
        adapter.load_state_dict(state)
        return adapter


class AttentiveNullSpaceAdapter(nn.Module):
    """Phase 2 adapter with attentive pooling over frame-level Conformer states."""

    def __init__(
        self,
        u_perp: torch.Tensor,
        conformer_dim: int = 1024,
        num_heads: int = 4,
        hidden_dim: int = 512,
        gamma_max: float = 1.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pool1 = AttentiveStatsPool(dim=conformer_dim, num_heads=num_heads)
        self.pool2 = AttentiveStatsPool(dim=conformer_dim, num_heads=num_heads)
        self.mlp = NullSpaceMLPAdapter(
            u_perp=u_perp,
            d_x1=2 * conformer_dim,
            d_x2=2 * conformer_dim,
            hidden_dim=hidden_dim,
            gamma_max=gamma_max,
            dropout=dropout,
            use_input_gate=True,
        )

    def forward(
        self,
        v_base: torch.Tensor,
        h_layer_a: torch.Tensor,
        h_layer_b: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x1 = self.pool1(h_layer_a, mask=mask)
        x2 = self.pool2(h_layer_b, mask=mask)
        return self.mlp(v_base=v_base, x1=x1, x2=x2)


def supervised_infonce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Multi-positive InfoNCE loss over L2-normalized embeddings."""
    emb = F.normalize(embeddings, p=2, dim=-1)
    sim = (emb @ emb.T) / temperature
    b = emb.shape[0]
    eye = torch.eye(b, dtype=torch.bool, device=emb.device)

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return emb.new_tensor(0.0)

    # Mask self-similarity out of denominator
    sim_masked = sim.masked_fill(eye, float("-inf"))
    log_probs = F.log_softmax(sim_masked, dim=1)
    pos_log_probs = (
        log_probs.masked_fill(~pos_mask, 0.0).sum(dim=1)
        / pos_mask.float().sum(dim=1).clamp(min=1.0)
    )
    return -pos_log_probs[has_pos].mean()


def semantic_preservation_loss(
    v_final: torch.Tensor,
    v_base: torch.Tensor,
) -> torch.Tensor:
    """Penalize drift of pairwise cosine similarities relative to v_base."""
    s_final = F.normalize(v_final, p=2, dim=-1) @ F.normalize(v_final, p=2, dim=-1).T
    s_base = F.normalize(v_base, p=2, dim=-1) @ F.normalize(v_base, p=2, dim=-1).T
    return F.mse_loss(s_final, s_base)
