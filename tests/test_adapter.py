import torch
import torch.nn.functional as F

from nullspace_audio import (
    AttentiveNullSpaceAdapter,
    NullSpaceMLPAdapter,
    compute_null_space_basis,
    semantic_preservation_loss,
    supervised_infonce_loss,
)


def test_mlp_adapter_preserves_text_orthogonality_and_roundtrips(tmp_path) -> None:
    torch.manual_seed(0)
    w = torch.randn(768, 512)
    u_perp = compute_null_space_basis(w)
    adapter = NullSpaceMLPAdapter(u_perp=u_perp)

    x1 = F.normalize(torch.randn(8, 2048), p=2, dim=-1)
    x2 = F.normalize(torch.randn(8, 1024), p=2, dim=-1)
    v_base = F.normalize(torch.randn(8, 512) @ w.T, p=2, dim=-1)

    shift, gamma = adapter.compute_shift(x1, x2)
    assert shift.shape == (8, 768)
    assert gamma.shape == (8, 1)

    # Shift is strictly orthogonal to every text embedding in col(W)
    text_embs = F.normalize(torch.randn(16, 512) @ w.T, p=2, dim=-1)
    dots = text_embs @ shift.T
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)

    out = adapter(v_base, x1, x2)
    assert out.shape == (8, 768)
    assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-5)

    path = tmp_path / "adapter.safetensors"
    adapter.save_safetensors(path)
    loaded = NullSpaceMLPAdapter.from_safetensors(path)
    loaded.eval()
    adapter.eval()
    assert torch.allclose(adapter(v_base, x1, x2), loaded(v_base, x1, x2), atol=1e-6)


def test_attentive_adapter_and_losses_backward() -> None:
    torch.manual_seed(1)
    w = torch.randn(768, 512)
    u_perp = compute_null_space_basis(w)
    adapter = AttentiveNullSpaceAdapter(u_perp=u_perp, conformer_dim=1024, num_heads=4)

    v_base = F.normalize(torch.randn(6, 512) @ w.T, p=2, dim=-1)
    h_a = torch.randn(6, 20, 1024)
    h_b = torch.randn(6, 20, 1024)
    mask = torch.ones(6, 20, dtype=torch.long)
    mask[0, 12:] = 0
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    v_final = adapter(v_base, h_a, h_b, mask=mask)
    loss = supervised_infonce_loss(v_final, labels) + 0.5 * semantic_preservation_loss(v_final, v_base)
    assert torch.isfinite(loss)
    loss.backward()

    # Check gradients flow to adapter parameters while u_perp remains a frozen buffer
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert has_grad
