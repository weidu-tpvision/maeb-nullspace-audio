from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .adapter import NullSpaceMLPAdapter
from .pca_head import NullSpaceAudioHead
from .pooling import masked_mean_pool, masked_mean_std_pool


def _find_conformer_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the 12-layer Conformer encoder ModuleList inside EmbeddingGemma-2."""
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == 12:
            if "audio" in name.lower() or "conformer" in name.lower():
                return module
    # Fallback: look for any 12-layer ModuleList whose children have Conformer-like names
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == 12:
            cls_name = module[0].__class__.__name__.lower()
            if "conformer" in cls_name or "audio" in cls_name:
                return module
    raise RuntimeError(
        "Could not locate 12-layer Conformer ModuleList in the base model. "
        "Pass layer modules explicitly or inspect model.named_modules()."
    )


class EmbeddingGemma2NullSpaceWrapper(nn.Module):
    """Wraps google/embeddinggemma-2 with a frozen 256D null-space head or trained adapter."""

    def __init__(
        self,
        base_model: nn.Module,
        processor: Any,
        head: NullSpaceAudioHead | NullSpaceMLPAdapter,
        gamma: float | None = 1.0,
        layer_a: int = 2,
        layer_b: int = 6,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.head = head
        self.gamma = gamma
        self.layer_a = layer_a
        self.layer_b = layer_b

        self.conformer_layers = _find_conformer_layers(base_model)
        self._captured: dict[int, torch.Tensor] = {}
        self._hooks: list[Any] = []

    @classmethod
    def from_pretrained(
        cls,
        head_path: str | Path,
        head_type: str = "pca",
        model_id: str = "google/embeddinggemma-2",
        revision: str | None = "914f7f89142e33e77833254d9c9b90c3cef7303b",
        gamma: float | None = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> EmbeddingGemma2NullSpaceWrapper:
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        base_model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
        ).to(device)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        if head_type == "pca":
            head: NullSpaceAudioHead | NullSpaceMLPAdapter = NullSpaceAudioHead.from_safetensors(head_path).to(device)
            if gamma is None:
                gamma = 1.0
        elif head_type == "mlp":
            head = NullSpaceMLPAdapter.from_safetensors(head_path).to(device)
        else:
            raise ValueError(f"Unknown head_type {head_type!r}; expected 'pca' or 'mlp'.")

        head.eval()
        return cls(base_model=base_model, processor=processor, head=head, gamma=gamma)

    def _register_hooks(self) -> None:
        self._captured.clear()

        def _make_hook(layer_idx: int):
            def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                self._captured[layer_idx] = tensor.detach()

            return _hook

        self._hooks.append(self.conformer_layers[self.layer_a].register_forward_hook(_make_hook(self.layer_a)))
        self._hooks.append(self.conformer_layers[self.layer_b].register_forward_hook(_make_hook(self.layer_b)))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.inference_mode()
    def encode_text(self, texts: list[str], **kwargs: Any) -> torch.Tensor:
        """Encode text identically to google/embeddinggemma-2 (no null-space shift)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, **kwargs)
        inputs = {k: v.to(self.head.u_perp.device) for k, v in inputs.items()}
        outputs = self.base_model(**inputs)
        emb = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs[0]
        return torch.nn.functional.normalize(emb.to(torch.float32), p=2, dim=-1)

    @torch.inference_mode()
    def encode_audio(
        self,
        audio_arrays: list[Any],
        sampling_rate: int = 16000,
        gamma: float | None = None,
    ) -> torch.Tensor:
        """Encode audio clips and inject the frozen 256D null-space acoustic shift."""
        active_gamma = self.gamma if gamma is None else gamma
        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        device = self.head.u_perp.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self._register_hooks()
        try:
            outputs = self.base_model(**inputs)
        finally:
            self._remove_hooks()

        v_base = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs[0]
        h_a = self._captured[self.layer_a]
        h_b = self._captured[self.layer_b]

        audio_mask = inputs.get("audio_attention_mask")
        if audio_mask is not None and audio_mask.shape[1] != h_a.shape[1]:
            # Downsampled Conformer frame mask if processor mask is at raw feature rate
            audio_mask = None

        x1 = masked_mean_std_pool(h_a, mask=audio_mask, l2_normalize=True)
        x2 = masked_mean_pool(h_b, mask=audio_mask, l2_normalize=True)
        return self.head(v_base=v_base, x1=x1, x2=x2, gamma=active_gamma)
