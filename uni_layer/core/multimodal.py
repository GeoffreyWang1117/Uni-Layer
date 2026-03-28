"""
Multi-modal model branch analysis.

Supports vision-language models (CLIP, LLaVA, etc.) with separate
analysis per encoder/decoder branch and cross-modal comparison.

Automatically detects common multi-modal architectures:
- model.vision_model / model.text_model  (CLIP)
- model.vision_tower / model.language_model  (LLaVA)
- model.visual / model.transformer  (OpenCLIP)
- model.image_encoder / model.text_encoder  (custom)
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from uni_layer.utils.layer_utils import get_model_layers

# Common multi-modal branch naming conventions
_VISION_ATTRS = (
    "vision_model",
    "vision_tower",
    "visual",
    "image_encoder",
    "vision_encoder",
    "img_encoder",
    "vit",
)

_LANGUAGE_ATTRS = (
    "text_model",
    "language_model",
    "text_encoder",
    "transformer",
    "lm_head",
    "decoder",
    "llm",
)

_FUSION_ATTRS = (
    "mm_projector",
    "multi_modal_projector",
    "fusion",
    "cross_attention",
    "bridge",
    "connector",
)


class MultiModalBranchAnalyzer:
    """
    Analyze multi-modal models by separating and comparing branches.

    Automatically detects vision, language, and fusion components,
    extracts layers from each, and enables per-branch and cross-branch analysis.

    Example:
        >>> mm = MultiModalBranchAnalyzer(clip_model)
        >>> print(mm.branches)  # {'vision': ..., 'language': ..., 'fusion': ...}
        >>> vision_layers = mm.get_branch_layers('vision')
        >>> report = mm.branch_summary()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.branches: Dict[str, nn.Module] = {}
        self.branch_layers: Dict[str, OrderedDict] = {}
        self._detect_branches()

    def _detect_branches(self):
        """Auto-detect multi-modal branches in the model."""
        for attr in _VISION_ATTRS:
            module = getattr(self.model, attr, None)
            if module is not None and isinstance(module, nn.Module):
                self.branches["vision"] = module
                break

        for attr in _LANGUAGE_ATTRS:
            module = getattr(self.model, attr, None)
            if module is not None and isinstance(module, nn.Module):
                # Avoid classifying a simple LM head as the language branch
                if (
                    len(list(module.children())) >= 1
                    and sum(p.numel() for p in module.parameters()) > 100
                ):
                    self.branches["language"] = module
                    break

        for attr in _FUSION_ATTRS:
            module = getattr(self.model, attr, None)
            if module is not None and isinstance(module, nn.Module):
                self.branches["fusion"] = module
                break

        # Extract layers per branch
        for name, branch in self.branches.items():
            self.branch_layers[name] = get_model_layers(branch)

    @property
    def is_multimodal(self) -> bool:
        """Whether the model has multiple detected branches."""
        return len(self.branches) >= 2

    @property
    def branch_names(self) -> List[str]:
        """List of detected branch names."""
        return list(self.branches.keys())

    def get_branch_layers(self, branch_name: str) -> OrderedDict:
        """Get layers for a specific branch."""
        return self.branch_layers.get(branch_name, OrderedDict())

    def get_all_layers(self) -> OrderedDict:
        """Get all layers with branch prefix."""
        all_layers = OrderedDict()
        for branch_name, layers in self.branch_layers.items():
            for layer_name, layer in layers.items():
                all_layers[f"{branch_name}/{layer_name}"] = layer
        return all_layers

    def branch_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary statistics per branch.

        Returns:
            Dict mapping branch name to {num_layers, num_params, param_ratio}
        """
        summary = {}
        total_params = sum(p.numel() for p in self.model.parameters())

        for name, branch in self.branches.items():
            branch_params = sum(p.numel() for p in branch.parameters())
            num_layers = len(self.branch_layers.get(name, {}))

            summary[name] = {
                "num_layers": num_layers,
                "num_params": branch_params,
                "param_ratio": branch_params / total_params if total_params > 0 else 0.0,
                "module_class": branch.__class__.__name__,
            }

        return summary

    def compare_branches(
        self,
        contributions_a: Dict[str, Dict[str, float]],
        contributions_b: Dict[str, Dict[str, float]],
        metric_name: str,
    ) -> Dict[str, float]:
        """
        Compare metric distributions between two branches.

        Args:
            contributions_a: compute_metrics() output for branch A
            contributions_b: compute_metrics() output for branch B
            metric_name: Metric key to compare

        Returns:
            Dict with mean_a, mean_b, ratio, divergence
        """
        values_a = [
            m[metric_name]
            for m in contributions_a.values()
            if metric_name in m and m[metric_name] is not None
        ]
        values_b = [
            m[metric_name]
            for m in contributions_b.values()
            if metric_name in m and m[metric_name] is not None
        ]

        if not values_a or not values_b:
            return {"mean_a": 0.0, "mean_b": 0.0, "ratio": 0.0, "kl_divergence": 0.0}

        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)

        # Ratio
        ratio = mean_a / mean_b if mean_b > 1e-10 else 0.0

        # Simple KL-like divergence (normalize to distributions)
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)
        # Normalize
        p = np.abs(arr_a) / (np.abs(arr_a).sum() + 1e-10)
        q = np.abs(arr_b) / (np.abs(arr_b).sum() + 1e-10)
        # Symmetric KL
        kl = 0.5 * np.sum(p * np.log((p + 1e-10) / (q.mean() + 1e-10))) + 0.5 * np.sum(
            q * np.log((q + 1e-10) / (p.mean() + 1e-10))
        )

        return {
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "ratio": float(ratio),
            "kl_divergence": float(kl),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export branch analysis as JSON-serializable dict."""
        return {
            "is_multimodal": self.is_multimodal,
            "branches": self.branch_names,
            "summary": self.branch_summary(),
            "layer_counts": {name: len(layers) for name, layers in self.branch_layers.items()},
        }
