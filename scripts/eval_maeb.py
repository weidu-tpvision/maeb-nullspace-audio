"""Evaluate the null-space audio wrapper on MAEB tasks via MTEB.

Usage:
    uv run python scripts/eval_maeb.py --head nullspace_audio_head.safetensors --gamma 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mteb
import numpy as np

from nullspace_audio import EmbeddingGemma2NullSpaceWrapper


class MTEBNullSpaceAdapter:
    """Minimal MTEB encoder adapter for EmbeddingGemma2NullSpaceWrapper."""

    def __init__(self, wrapper: EmbeddingGemma2NullSpaceWrapper) -> None:
        self.wrapper = wrapper

    def encode(self, sentences, **kwargs):
        return self.wrapper.encode_text(list(sentences)).cpu().numpy()

    def get_audio_embeddings(self, audio_batches, sampling_rate: int = 16000, **kwargs):
        embs = []
        for batch in audio_batches:
            embs.append(self.wrapper.encode_audio(batch, sampling_rate=sampling_rate).cpu().numpy())
        return np.concatenate(embs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--head-type", choices=["pca", "mlp"], default="pca")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["IEMOCAPGender", "CREMA_D", "GTZANGenre", "VoxPopuliGenderClustering"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    wrapper = EmbeddingGemma2NullSpaceWrapper.from_pretrained(
        args.head,
        head_type=args.head_type,
        gamma=args.gamma,
    )
    adapter = MTEBNullSpaceAdapter(wrapper)
    tasks = mteb.get_tasks(tasks=args.tasks)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(adapter, output_folder=str(args.output_dir))
    print(results)


if __name__ == "__main__":
    main()
