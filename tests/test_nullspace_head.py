import numpy as np
import torch
import torch.nn.functional as F

from nullspace_audio import (
    NullSpaceAudioHead,
    compute_null_space_basis,
    fit_balanced_pca,
    masked_mean_pool,
    masked_mean_std_pool,
)


def test_null_space_orthogonality_and_zero_text_interference() -> None:
    torch.manual_seed(0)
    # Simulate EmbeddingGemma-2's terminal linear projection W: [768, 512]
    w = torch.randn(768, 512)
    u_perp = compute_null_space_basis(w)

    assert u_perp.shape == (768, 256)
    # Orthonormality: U_perp^T @ U_perp == I_256
    gram = u_perp.T @ u_perp
    assert torch.allclose(gram, torch.eye(256), atol=1e-5)

    # Zero dot product with any text embedding y = W @ x
    text_backbone = torch.randn(32, 512)
    text_embs = F.normalize(text_backbone @ w.T, p=2, dim=-1)
    z = torch.randn(16, 256)
    shifts = z @ u_perp.T

    dots = text_embs @ shifts.T
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)


def test_masked_pooling_ignores_padded_frames() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(2, 10, 64)
    # Pad last 4 frames of item 0 and last 2 frames of item 1 with huge garbage values
    hidden[0, 6:, :] = 1e6
    hidden[1, 8:, :] = -1e6

    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )

    m = masked_mean_pool(hidden, mask=mask, l2_normalize=False)
    ms = masked_mean_std_pool(hidden, mask=mask, l2_normalize=False)

    assert torch.allclose(m[0], hidden[0, :6].mean(dim=0), atol=1e-5)
    assert torch.allclose(m[1], hidden[1, :8].mean(dim=0), atol=1e-5)
    assert ms.shape == (2, 128)
    assert torch.isfinite(ms).all()


def test_balanced_pca_soft_whiten_and_safetensors_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(42)
    domains_x1 = [rng.normal(size=(40, 2048)) for _ in range(4)]
    domains_x2 = [rng.normal(size=(40, 1024)) for _ in range(4)]

    pca1 = fit_balanced_pca(domains_x1, n_components=128)
    pca2 = fit_balanced_pca(domains_x2, n_components=128)

    w = torch.randn(768, 512)
    u_perp = compute_null_space_basis(w)
    head = NullSpaceAudioHead(pca1=pca1, pca2=pca2, u_perp=u_perp)

    path = tmp_path / "head.safetensors"
    head.save_safetensors(path)
    loaded = NullSpaceAudioHead.from_safetensors(path)

    v_base = F.normalize(torch.randn(4, 768), p=2, dim=-1)
    x1 = F.normalize(torch.randn(4, 2048), p=2, dim=-1)
    x2 = F.normalize(torch.randn(4, 1024), p=2, dim=-1)

    out1 = head(v_base, x1, x2, gamma=1.0)
    out2 = loaded(v_base, x1, x2, gamma=1.0)
    assert out1.shape == (4, 768)
    assert torch.allclose(out1, out2, atol=1e-6)
    # Output is L2-normalized
    norms = out1.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)
