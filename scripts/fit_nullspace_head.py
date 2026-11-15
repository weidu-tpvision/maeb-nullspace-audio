"""Fit the zero-leakage 256D null-space audio head on 4 external audio corpora.

Usage:
    uv run python scripts/fit_nullspace_head.py --output nullspace_audio_head.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoModel, AutoProcessor

from nullspace_audio import (
    NullSpaceAudioHead,
    compute_null_space_basis,
    find_terminal_projection,
    fit_balanced_pca,
    masked_mean_pool,
    masked_mean_std_pool,
)
from nullspace_audio.wrapper import _find_conformer_layers

EXTERNAL_CORPORA = [
    ("ylacombe/expresso", "default", "train"),
    ("agkphysics/AudioSet", "balanced", "train"),
    ("lewtun/music_genres", "default", "train"),
    ("ashraq/esc50", "default", "train"),
]


def load_audio_clips(
    dataset_id: str,
    config: str,
    split: str,
    max_clips: int,
    max_seconds: float,
    target_sr: int = 16000,
) -> list[np.ndarray]:
    ds = load_dataset(dataset_id, config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=target_sr))
    clips: list[np.ndarray] = []
    max_samples = int(max_seconds * target_sr)

    for row in ds:
        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if sr != target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if len(arr) > max_samples:
            arr = arr[:max_samples]
        if len(arr) >= target_sr // 4:  # skip tiny/empty clips (< 0.25s)
            clips.append(arr)
        if len(clips) >= max_clips:
            break
    return clips


@torch.inference_mode()
def extract_layer_features(
    model: torch.nn.Module,
    processor,
    clips: list[np.ndarray],
    layer_a: int,
    layer_b: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    conformer_layers = _find_conformer_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def _hook(_mod, _inp, out):
            captured[idx] = (out[0] if isinstance(out, tuple) else out).detach()

        return _hook

    h1 = conformer_layers[layer_a].register_forward_hook(_make_hook(layer_a))
    h2 = conformer_layers[layer_b].register_forward_hook(_make_hook(layer_b))

    x1_all, x2_all = [], []
    try:
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            inputs = processor(audio=batch, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
            x1 = masked_mean_std_pool(captured[layer_a], l2_normalize=True).cpu().numpy()
            x2 = masked_mean_pool(captured[layer_b], l2_normalize=True).cpu().numpy()
            x1_all.append(x1)
            x2_all.append(x2)
    finally:
        h1.remove()
        h2.remove()

    return np.concatenate(x1_all, axis=0), np.concatenate(x2_all, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/embeddinggemma-2")
    parser.add_argument("--revision", default="914f7f89142e33e77833254d9c9b90c3cef7303b")
    parser.add_argument("--clips-per-corpus", type=int, default=350)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--layer-a", type=int, default=2)
    parser.add_argument("--layer-b", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("nullspace_audio_head.safetensors"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    w_term = find_terminal_projection(model, out_dim=768, in_dim=512)
    u_perp = compute_null_space_basis(w_term)
    print(f"Extracted null-space basis U_perp with shape {tuple(u_perp.shape)}")

    domain_x1, domain_x2 = [], []
    for ds_id, cfg, split in EXTERNAL_CORPORA:
        print(f"Streaming up to {args.clips_per_corpus} clips from {ds_id}...")
        clips = load_audio_clips(ds_id, cfg, split, args.clips_per_corpus, args.max_seconds)
        x1, x2 = extract_layer_features(
            model, processor, clips, args.layer_a, args.layer_b, args.batch_size, device
        )
        domain_x1.append(x1)
        domain_x2.append(x2)
        print(f"  {ds_id}: {len(clips)} clips -> X1 {x1.shape}, X2 {x2.shape}")

    pca1 = fit_balanced_pca(domain_x1, n_components=128)
    pca2 = fit_balanced_pca(domain_x2, n_components=128)

    head = NullSpaceAudioHead(pca1=pca1, pca2=pca2, u_perp=u_perp)
    head.save_safetensors(args.output)
    print(f"Saved frozen null-space audio head to {args.output}")


if __name__ == "__main__":
    main()
