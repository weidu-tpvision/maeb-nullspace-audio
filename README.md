# MAEB Null-Space Audio

Reimplementation of Bruno Hays' zero-training null-space audio embedding method ([blog post](https://bruno-hays.github.io/posts/maeb-null-space-audio/), [model](https://huggingface.co/BrunoHays/embeddinggemma-2-nullspace-audio)).

## Core Idea

1. **Extract the 256D null space** of `google/embeddinggemma-2`'s terminal linear projection $W \in \mathbb{R}^{768 \times 512}$ via SVD ($U_\perp \in \mathbb{R}^{768 \times 256}$). Any vector in $\text{span}(U_\perp)$ has zero dot product with all text embeddings.
2. **Pool intermediate Conformer layers** from the frozen audio encoder:
   - Layer 2: masked mean + std over time $\rightarrow 2048\text{D}$ (L2-normalized)
   - Layer 6: masked mean over time $\rightarrow 1024\text{D}$ (L2-normalized)
3. **Fit balanced, soft-whitened PCA** on 1,400 external audio clips (350 each from `ylacombe/expresso`, `agkphysics/AudioSet`, `lewtun/music_genres`, `ashraq/esc50`, capped at 10s), reducing each block to $128\text{D}$ ($256\text{D}$ total).
4. **Inject into the null space** at inference time:
   $$v_{\text{final}} = \text{normalize}\big(v_{\text{base}} + \gamma \cdot (z \, U_\perp^T)\big)$$

## Setup

```bash
uv sync --extra dev
# For MAEB benchmark evaluation:
uv sync --extra eval --extra dev
```

You must accept the Gemma license terms on Hugging Face for [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) and authenticate with `hf auth login` before downloading weights.

## Usage

### 1. Run unit tests (verifies math, orthogonality, pooling, and safetensors I/O)

```bash
uv run pytest
```

### 2. Fit the null-space head on the 4 external corpora

```bash
uv run python scripts/fit_nullspace_head.py --output nullspace_audio_head.safetensors
```

### 3. Probe individual Conformer layers

```bash
uv run python scripts/probe_layers.py --dataset ashraq/esc50 --max-clips 400
```

### 4. Evaluate on MAEB

```bash
uv run python scripts/eval_maeb.py --head nullspace_audio_head.safetensors --gamma 1.0
```
