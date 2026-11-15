"""Train a Phase 1 NullSpaceMLPAdapter on cached Conformer feature shards.

Expected shard format (.pt dict):
    {
        "v_base": Tensor[N, 768],
        "x1": Tensor[N, 2048],      # Layer 2 masked [mean; std]
        "x2": Tensor[N, 1024],      # Layer 6 masked mean
        "labels": Tensor[N],        # integer class/speaker/cluster labels
        "u_perp": Tensor[768, 256], # SVD null-space basis of terminal W
    }

For a quick end-to-end smoke test without cached data:
    uv run python scripts/train_adapter.py --synthetic-smoke-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from nullspace_audio import (
    NullSpaceMLPAdapter,
    compute_null_space_basis,
    semantic_preservation_loss,
    supervised_infonce_loss,
)


def make_synthetic_dataset(n_samples: int = 1024, n_classes: int = 16) -> tuple[TensorDataset, torch.Tensor]:
    torch.manual_seed(42)
    w = torch.randn(768, 512)
    u_perp = compute_null_space_basis(w)

    labels = torch.randint(0, n_classes, (n_samples,))
    class_proto_x1 = torch.randn(n_classes, 2048)
    class_proto_x2 = torch.randn(n_classes, 1024)

    v_base = torch.nn.functional.normalize(torch.randn(n_samples, 512) @ w.T, p=2, dim=-1)
    x1 = torch.nn.functional.normalize(class_proto_x1[labels] + 0.3 * torch.randn(n_samples, 2048), p=2, dim=-1)
    x2 = torch.nn.functional.normalize(class_proto_x2[labels] + 0.3 * torch.randn(n_samples, 1024), p=2, dim=-1)
    return TensorDataset(v_base, x1, x2, labels), u_perp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-pt", type=Path, default=None, help="Path to cached .pt feature dict")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-sem", type=float, default=0.5)
    parser.add_argument("--lambda-reg", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=Path("nullspace_mlp_adapter.safetensors"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.synthetic_smoke_test:
        dataset, u_perp = make_synthetic_dataset()
    elif args.features_pt is not None:
        blob = torch.load(args.features_pt, map_location="cpu", weights_only=True)
        u_perp = blob["u_perp"]
        dataset = TensorDataset(blob["v_base"], blob["x1"], blob["x2"], blob["labels"])
    else:
        raise ValueError("Provide --features-pt <path> or --synthetic-smoke-test")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    adapter = NullSpaceMLPAdapter(u_perp=u_perp).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(loader)))

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss = 0.0
        for v_b, x1, x2, labels in loader:
            v_b, x1, x2, labels = v_b.to(device), x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            shift, _ = adapter.compute_shift(x1, x2)
            v_final = torch.nn.functional.normalize(
                torch.nn.functional.normalize(v_b, p=2, dim=-1) + shift, p=2, dim=-1
            )

            loss_nce = supervised_infonce_loss(v_final, labels)
            loss_sem = semantic_preservation_loss(v_final, v_b)
            loss_reg = (shift ** 2).sum(dim=-1).mean()
            loss = loss_nce + args.lambda_sem * loss_sem + args.lambda_reg * loss_reg

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg = total_loss / max(1, len(loader))
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | loss={avg:.4f}")

    adapter.save_safetensors(args.output)
    print(f"Saved trained NullSpaceMLPAdapter to {args.output}")


if __name__ == "__main__":
    main()
