# Plan: Extending Null-Space PCA to a Trained Null-Space Adapter

## 1. Feasibility Assessment

**Feasibility: Very High (low compute, zero risk to text embeddings).**

Replacing the unsupervised PCA projection with a gradient-trained adapter/projector is straightforward and inexpensive for three reasons:

1. **Frozen 577M Backbone:** `google/embeddinggemma-2` stays 100% frozen. Gradients only flow through a lightweight adapter (~1M–5M parameters) attached to Conformer Layer 2 and Layer 6 (or a learned weighted sum of all 12 layers).
2. **Hard Orthogonality Guarantee Preserved:** Because the adapter's 256D output $z$ is still projected through the fixed SVD null-space basis $U_\perp \in \mathbb{R}^{768 \times 256}$:
   $$v_{\text{final}} = \text{normalize}\big(v_{\text{base}} + \gamma(x) \cdot (z \, U_\perp^T)\big)$$
   every text embedding remains **bit-for-bit identical**, and text-to-audio dot products before normalization are strictly unchanged.
3. **Fast Iteration:** If operating on cached pooled layer features (`X1: 2048D`, `X2: 1024D`), training takes **minutes on a single GPU** (or even CPU). With frame-level attentive pooling over `[B, T, 1024]` Conformer states, training takes ~1–2 hours on a single L4/T4/3090 GPU.

### Main Risk & Mitigation
- **Risk:** Overfitting to specific training corpora or worsening the semantic audio-to-audio regressions seen with PCA (e.g., SIB-FLEURS, MInDS14, NMSQA).
- **Mitigation:**
  - Strictly exclude MAEB/MSEB evaluation splits from training data.
  - Add a **semantic-preservation regularizer** (penalizing cosine drift on same-transcript/different-speaker pairs) and an **input-dependent gate** $\gamma(x) \in [0, \gamma_{\max}]$.

---

## 2. Architecture Roadmap

### Phase 1: Cached-Feature MLP Projector (Fastest Baseline)
Replace the two 128D PCA blocks with a lightweight residual MLP operating on the same pooled features (`X1` from Layer 2 `[mean; std]`, `X2` from Layer 6 `mean`):
- `Proj1`: `Linear(2048, 512) -> LayerNorm -> GELU -> Dropout(0.1) -> Linear(512, 128)`
- `Proj2`: `Linear(1024, 512) -> LayerNorm -> GELU -> Dropout(0.1) -> Linear(512, 128)`
- Concatenate `[w1 * z1, w2 * z2]` (with learnable block weights `w1, w2`), L2-normalize to `z` (`256D`), and project via `z @ U_perp.T`.

### Phase 2: Attentive Pooling + Layer Mixing + Learned Gate
Uniform `mean`/`std` pooling weights silence and background noise equally with voiced frames. Upgrade to:
1. **Scalar Layer Mixing (optional):** Learn softmax weights $\alpha_0, \dots, \alpha_{11}$ over all 12 Conformer layers instead of hardcoding Layers 2 and 6.
2. **Multi-Head Attentive Statistics Pooling:** Attend over `[B, T, 1024]` frame states to extract weighted mean and std focused on salient acoustic/prosodic frames.
3. **Adaptive Gating $\gamma(x)$:** Predict a bounded scalar $\gamma(x) = \gamma_{\max} \cdot \sigma(\text{Linear}([X_1; X_2]))$ so the adapter can down-weight acoustic injection when a clip is primarily linguistic/low-paralinguistic.

---

## 3. Training Data & Multi-Task Objective

### Dataset Scale Requirements (Why 1,400 Clips Is Not Enough)
While 1,400 clips (350 per corpus) was sufficient for unsupervised PCA, gradient-based adapters have far more degrees of freedom and will overfit small labeled datasets:

| Regime | Dataset Size | Recommended Architecture |
|---|---|---|
| Tiny | `1.4k–5k` clips | Keep PCA frozen; only train a **256D per-dim weight + $\gamma(x)$ gate** (~1k params) |
| Medium | `10k–20k` clips | **Linear projector** (`2048→128`, `1024→128`, ~390k params) with strong weight decay |
| Full | `50k–200k+` clips | **Phase 1 MLP** or **Phase 2 Attentive Adapter** (1M–5M params) |

Because the 577M backbone stays frozen, caching features for 100k+ clips requires only a single inference pass.

### External Training Corpora (Zero MAEB Leakage)
| Target Signal | Candidate External Datasets |
|---|---|
| Speaker identity / gender / pitch | VoxCeleb2 (dev), LibriSpeech, Expresso |
| Vocal emotion / prosody | Expresso, MSP-Podcast (non-MAEB splits) |
| Environmental sounds & audio events | AudioSet (balanced), FSD50K (dev) |
| Music genre / artist / timbre | MTG-Jamendo (train split, disjoint from eval), FMA |

### Loss Function
Train with a multi-task contrastive + semantic-preservation objective:

$$\mathcal{L} = \mathcal{L}_{\text{InfoNCE}}(\text{paralinguistic}) + \lambda_{\text{sem}} \, \mathcal{L}_{\text{sem\_preserve}} + \lambda_{\text{reg}} \, \|\gamma(x) \, z\|_2^2$$

1. **$\mathcal{L}_{\text{InfoNCE}}$ (Supervised / Multi-Positive Contrastive):**
   - Pull together clips sharing the same speaker, emotion, sound class, or music genre/artist; push apart negatives in-batch.
   - Compute cosine similarity directly on $v_{\text{final}}$ so the adapter learns how to share the 768D hypersphere with $v_{\text{base}}$.
2. **$\mathcal{L}_{\text{sem\_preserve}}$ (Semantic Preservation Regularizer):**
   - On pairs with the same spoken content but different speakers/prosody (or augmented speed/pitch pairs), penalize excessive drift from `v_base` cosine similarity. This directly targets the SIB-FLEURS / MInDS14 / NMSQA regression.
3. **$\|\gamma(x) z\|_2^2$ (Norm Regularizer):**
   - Prevents the null-space vector from growing large enough to shrink the semantic component $v_{\text{base}}$ during L2 normalization.

---

## 4. Implementation Steps in This Repo

1. **Feature Caching (`scripts/cache_features.py`):**
   - Stream external datasets, run frozen `google/embeddinggemma-2` once, and save `(v_base, x1, x2, labels)` shards to disk (or frame-level Conformer states for Phase 2).
2. **Trainable Module (`src/nullspace_audio/adapter.py`):**
   - Implement `NullSpaceMLPAdapter` (Phase 1) and `AttentiveNullSpaceAdapter` (Phase 2), exporting trained weights to `.safetensors` compatible with `EmbeddingGemma2NullSpaceWrapper`.
3. **Training Script (`scripts/train_adapter.py`):**
   - AdamW (`lr=3e-4`, cosine schedule, weight decay `1e-2`), mixed precision (`bf16`), batch size 128–256 for strong in-batch contrastive negatives.
4. **Evaluation (`scripts/eval_maeb.py`):**
   - Benchmark on all 30 MAEB tasks + MSEB SVQ.
   - Track two success criteria:
     1. Higher macro score on paralinguistic/acoustic tasks than PCA (`> 55.09%`).
     2. Smaller regression on semantic audio tasks (SIB-FLEURS, MInDS14, NMSQA) compared to PCA.
