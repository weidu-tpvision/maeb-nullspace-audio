from __future__ import annotations

import torch
import torch.nn as nn


def compute_null_space_basis(
    weight: torch.Tensor,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Compute an orthonormal basis U_perp for the left null space of W.

    Given a linear projection matrix W of shape [out_dim, in_dim] (e.g. [768, 512])
    that maps from in_dim -> out_dim without bias/activation, every output y = W @ x
    lies in col(W). The left null space U_perp spans the orthogonal complement
    of col(W), so W^T @ U_perp == 0 and y^T @ U_perp == 0 for all x.

    Args:
        weight: Projection matrix of shape [out_dim, in_dim] with out_dim > in_dim.
        tol: Singular-value threshold relative to max singular value for rank detection.

    Returns:
        U_perp of shape [out_dim, out_dim - rank] with U_perp^T @ U_perp = I.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    if out_dim <= in_dim:
        raise ValueError(
            f"Expected out_dim > in_dim to have a non-trivial left null space, "
            f"got {tuple(weight.shape)}"
        )

    w_f64 = weight.detach().to(dtype=torch.float64, device="cpu")
    u, s, _ = torch.linalg.svd(w_f64, full_matrices=True)
    rank = int((s > tol * s[0]).sum().item())
    u_perp = u[:, rank:].contiguous().to(dtype=torch.float32)
    return u_perp


def find_terminal_projection(
    model: nn.Module,
    out_dim: int = 768,
    in_dim: int = 512,
) -> torch.Tensor:
    """Locate the terminal linear projection W of shape [out_dim, in_dim] in a model.

    Searches modules in reverse order for an nn.Linear with weight shape [out_dim, in_dim].
    """
    candidates: list[tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and tuple(module.weight.shape) == (out_dim, in_dim):
            candidates.append((name, module.weight))

    if not candidates:
        raise RuntimeError(
            f"Could not find an nn.Linear layer with weight shape ({out_dim}, {in_dim}) in model."
        )

    # The terminal projection is the last such linear layer in the module tree.
    _, weight = candidates[-1]
    return weight.detach().clone()
