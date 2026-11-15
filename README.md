# MAEB Null-Space Audio

Reimplementation of Bruno Hays' zero-training null-space audio embedding method ([blog post](https://bruno-hays.github.io/posts/maeb-null-space-audio/), [model](https://huggingface.co/BrunoHays/embeddinggemma-2-nullspace-audio)), plus an experimental gradient-trained adapter extension for learning and comparison.

## Core Idea

1. **Extract the 256D null space** of `google/embeddinggemma-2`'s terminal linear projection $W \in \mathbb{R}^{768 \times 512}$ via SVD ($U_\perp \in \mathbb{R}^{768 \times 256}$). Any vector in $\text{span}(U_\perp)$ has zero dot product with all text embeddings.
2. **Pool intermediate Conformer layers** from the frozen audio encoder:
   - Layer 2: masked mean + std over time $\rightarrow 2048\text{D}$ (L2-normalized)
   - Layer 6: masked mean over time $\rightarrow 1024\text{D}$ (L2-normalized)
3. **Fit balanced, soft-whitened PCA** on 1,400 external audio clips (350 each from `ylacombe/expresso`, `agkphysics/AudioSet`, `lewtun/music_genres`, `ashraq/esc50`, capped at 10s), reducing each block to $128\text{D}$ ($256\text{D}$ total).
4. **Inject into the null space** at inference time:
   $$v_{\text{final}} = \text{normalize}\big(v_{\text{base}} + \gamma \cdot (z \, U_\perp^T)\big)$$

## Why PCA vs. a Trained Adapter?

- **Why the original post used PCA:** The goal of Bruno Hays' experiment was a **zero-training proof of concept**—showing that speaker, pitch, and emotion signals are *already* sitting in the frozen intermediate Conformer layers and can be recovered without gradient updates, labeled datasets, or the risk of overfitting to benchmark tasks.
- **Why this repo also includes a trained adapter:** Strictly **for fun, learning, and comparison**. While PCA is unsupervised (variance ≠ task usefulness) and unweighted (which hurts purely semantic speech tasks like SIB-FLEURS or MInDS14), a lightweight gradient-trained adapter (`src/nullspace_audio/adapter.py`) lets you explore:
  - Learned non-linear projections (`NullSpaceMLPAdapter`) and attentive temporal pooling (`AttentiveNullSpaceAdapter`),
  - Input-dependent gating $\gamma(x)$ and a semantic-preservation regularizer to reduce regressions on linguistic audio tasks,
  - All while keeping the **exact same frozen backbone and $U_\perp$ null-space constraint** so text embeddings remain bit-for-bit identical.

See [`docs/ADAPTER_PLAN.md`](docs/ADAPTER_PLAN.md) for the full adapter roadmap and dataset-scale guidelines (including why 1,400 clips is enough for PCA but 50k–200k+ clips are needed for a multi-task MLP/attentive adapter).

## Setup

```bash
uv sync --extra dev
# For MAEB benchmark evaluation:
uv sync --extra eval --extra dev
```

You must accept the Gemma license terms on Hugging Face for [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) and authenticate with `hf auth login` before downloading weights.

## Usage

### 1. Run unit tests (verifies SVD orthogonality, pooling, PCA, and adapters)

```bash
uv run pytest -v
```

### 2. Zero-Training PCA Baseline

Fit the frozen null-space head on the 4 external corpora:

```bash
uv run python scripts/fit_nullspace_head.py --output nullspace_audio_head.safetensors
```

Probe individual Conformer layers:

```bash
uv run python scripts/probe_layers.py --dataset ashraq/esc50 --max-clips 400
```

Evaluate on MAEB:

```bash
uv run python scripts/eval_maeb.py --head nullspace_audio_head.safetensors --gamma 1.0
```

### 3. Trained Null-Space Adapter (Experimental)

Run a quick synthetic smoke test of the training loop:

```bash
uv run python scripts/train_adapter.py --synthetic-smoke-test --epochs 3
```

Train on cached Conformer feature shards (`.pt` dict with `v_base`, `x1`, `x2`, `labels`, `u_perp`):

```bash
uv run python scripts/train_adapter.py --features-pt /path/to/cached_features.pt --output nullspace_mlp_adapter.safetensors
```
