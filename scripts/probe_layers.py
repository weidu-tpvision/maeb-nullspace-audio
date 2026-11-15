"""Probe all 12 Conformer layers individually using linear classification (logistic regression).

Reproduces the layer-wise probing experiment showing that early/middle layers
(Layer 2 for speaker/acoustics, Layer 6 for emotion) retain paralinguistic traits
that disappear by the final layer.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import Audio, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoModel, AutoProcessor

from nullspace_audio.pooling import masked_mean_pool
from nullspace_audio.wrapper import _find_conformer_layers


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/embeddinggemma-2")
    parser.add_argument("--dataset", default="ashraq/esc50")
    parser.add_argument("--split", default="train")
    parser.add_argument("--label-col", default="category")
    parser.add_argument("--max-clips", type=int, default=400)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    layers = _find_conformer_layers(model)

    ds = load_dataset(args.dataset, split=args.split).cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.select(range(min(args.max_clips, len(ds))))

    clips = [np.asarray(r["audio"]["array"], dtype=np.float32)[:160000] for r in ds]
    labels = [r[args.label_col] for r in ds]

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for idx, layer in enumerate(layers):
        def _make_hook(i: int):
            return lambda _m, _in, out: captured.__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach()
            )
        hooks.append(layer.register_forward_hook(_make_hook(idx)))

    per_layer_feats: dict[int, list[np.ndarray]] = {i: [] for i in range(len(layers))}
    try:
        for start in range(0, len(clips), 8):
            batch = clips[start : start + 8]
            inputs = processor(audio=batch, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
            for idx in range(len(layers)):
                pooled = masked_mean_pool(captured[idx], l2_normalize=True).cpu().numpy()
                per_layer_feats[idx].append(pooled)
    finally:
        for h in hooks:
            h.remove()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"Probing {len(layers)} Conformer layers on {args.dataset} ({len(clips)} clips):")
    for idx in range(len(layers)):
        x = np.concatenate(per_layer_feats[idx], axis=0)
        clf = LogisticRegression(max_iter=1000)
        acc = cross_val_score(clf, x, labels, cv=cv, scoring="accuracy").mean()
        print(f"  Layer {idx:2d}: 5-fold CV accuracy = {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
