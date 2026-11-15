# MAEB Null-Space Audio

Reimplementation of Bruno Hays' zero-training null-space audio embedding method ([blog post](https://bruno-hays.github.io/posts/maeb-null-space-audio/), [model](https://huggingface.co/BrunoHays/embeddinggemma-2-nullspace-audio)), plus an experimental gradient-trained adapter extension for learning and comparison.

## Summary of the Original Blog Post

In [*Audio embeddings forget the voice: recovering it from intermediate layers takes EmbeddingGemma-2 to rank 1 on MAEB*](https://bruno-hays.github.io/posts/maeb-null-space-audio/), Bruno Hays demonstrates how to recover paralinguistic information from a frozen multimodal embedder **without training a single weight**:

- **The Problem:** Audio encoders aligned to a text LLM learn *invariance* to who is speaking and how they sound—keeping *what* was said (95–99% on multilingual text-to-audio retrieval) while discarding speaker identity, pitch, emotion, and sound texture by the final layer (0.05% on VoxPopuli gender clustering, 19.7% on CREMA-D emotion).
- **Where the Signal Lives:** Probing EmbeddingGemma-2's 12 Conformer layers individually reveals that **Layer 2** still holds speaker traits (~65% on VoxPopuli gender clustering) and **Layer 6** holds vocal emotion (39.5% on CREMA-D).
- **Why It Can Be Added for Free (256D Null Space):** EmbeddingGemma-2's text backbone is 512D, projected linearly to 768D by a terminal matrix $W \in \mathbb{R}^{768 \times 512}$ with no activation function. Because every text embedding lies strictly in $\text{col}(W)$, SVD yields **256 orthogonal directions** ($U_\perp$) that text never touches—so any vector injected along $U_\perp$ has a **zero dot product** with all text embeddings.
- **Headline Results:** Injecting PCA-compressed Layer 2 & 6 features into $U_\perp$ lifts the 30-task MAEB macro score from **49.3% → 54.4%** (Rank #16 → #1 overall, #26 → #1 audio-only), with major gains on VoxPopuli gender clustering (`0.05% → 63.1%`), Jamendo artist retrieval (`55.2% → 93.8%`), CREMA-D emotion (`19.7% → 40.2%`), and FSD2019 (`38.8% → 56.3%`).
- **Why Semantic Audio Tasks Regress:**
  - *Text-to-audio* is largely unaffected because $U_\perp$ is orthogonal to text.
  - On *audio-to-audio semantic tasks* (SIB-FLEURS `57.2% → 21.3%`, MInDS14 `97.6% → 88.3%`, NMSQA `98.1% → 78.7%`), **both** clips carry the unweighted 256D acoustic vector. Cosine similarity then blends semantic and acoustic similarity, pushing apart clips with the same intent spoken by different voices or emotions.
  - *No Matryoshka truncation for audio:* Because $U_\perp$ is spread across all 768 coordinates, truncating audio embeddings to 512/256/128D breaks the orthogonality and loses part of the acoustic signal.

## Core Idea (Pipeline)

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
