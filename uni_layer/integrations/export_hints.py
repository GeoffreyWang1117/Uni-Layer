"""
ONNX / TensorRT export optimization hints.

Generates per-layer optimization recommendations based on Uni-Layer analysis:
- Quantization sensitivity: which layers tolerate INT8/FP16 vs. need FP32
- Operator fusion candidates: adjacent layers that can be fused
- Layer pruning recommendations: layers safe to remove for inference
- TensorRT-specific hints: precision, workspace, kernel selection

These hints can be used to guide ONNX graph optimization and
TensorRT engine building for maximum inference throughput.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn


class ExportHintsBridge:
    """
    Generate ONNX/TensorRT optimization hints from Uni-Layer analysis.

    Uses layer contribution metrics to recommend:
    - Per-layer quantization precision (FP32 / FP16 / INT8)
    - Layers safe to prune for inference
    - Operator fusion opportunities
    - TensorRT builder configuration hints

    Example:
        >>> bridge = ExportHintsBridge(model, contributions)
        >>> hints = bridge.quantization_plan(target="int8", protect_top_k=5)
        >>> print(hints)  # {"layer.0": "int8", "layer.5": "fp32", ...}
    """

    def __init__(
        self,
        model: nn.Module,
        contributions: Dict[str, Dict[str, Any]],
    ):
        self.model = model
        self.contributions = contributions
        self.layer_names = list(contributions.keys())

    def quantization_plan(
        self,
        metric_name: str = "gradient_norm",
        target: str = "fp16",
        protect_ratio: float = 0.2,
        sensitivity_threshold: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Generate per-layer quantization precision recommendations.

        High-importance layers stay at higher precision (protected),
        low-importance layers can be aggressively quantized.

        Args:
            metric_name: Metric to rank importance by
            target: Target precision for non-critical layers ("fp16", "int8")
            protect_ratio: Fraction of top layers to keep at FP32
            sensitivity_threshold: If set, layers above this threshold stay FP32

        Returns:
            Dict mapping layer_name to precision string ("fp32", "fp16", "int8")
        """
        # Rank layers by importance
        ranked = self._rank_by_metric(metric_name)

        if not ranked:
            return {name: target for name in self.layer_names}

        num_protect = max(1, int(len(ranked) * protect_ratio))

        if sensitivity_threshold is not None:
            protected = {name for name, score in ranked if score >= sensitivity_threshold}
        else:
            protected = {name for name, _ in ranked[:num_protect]}

        plan = {}
        for name in self.layer_names:
            if name in protected:
                plan[name] = "fp32"
            else:
                plan[name] = target

        return plan

    def prunable_layers(
        self,
        metric_name: str = "block_influence",
        threshold: float = 0.02,
        max_ratio: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Identify layers safe to remove for inference optimization.

        Args:
            metric_name: Metric to judge importance
            threshold: Layers below this score are candidates
            max_ratio: Maximum fraction of layers to recommend removing

        Returns:
            List of {name, score, recommendation} dicts
        """
        ranked = self._rank_by_metric(metric_name, ascending=True)

        if not ranked:
            return []

        max_remove = max(1, int(len(ranked) * max_ratio))
        candidates = []

        for name, score in ranked:
            if score < threshold and len(candidates) < max_remove:
                candidates.append(
                    {
                        "layer_name": name,
                        "score": float(score),
                        "recommendation": "remove",
                    }
                )

        return candidates

    def fusion_candidates(self) -> List[Dict[str, Any]]:
        """
        Identify adjacent layer pairs that are candidates for operator fusion.

        Common fusion patterns:
        - Conv + BatchNorm + ReLU
        - Linear + ReLU
        - Linear + GELU
        - LayerNorm + Linear

        Returns:
            List of {layers, pattern, description} dicts
        """
        candidates = []
        names = self.layer_names

        for i in range(len(names) - 1):
            type_i = self.contributions[names[i]].get("layer_type", "")
            type_j = (
                self.contributions[names[i + 1]].get("layer_type", "") if i + 1 < len(names) else ""
            )

            # Detect fusion patterns
            if type_i == "convolution" and type_j == "normalization":
                candidates.append(
                    {
                        "layers": [names[i], names[i + 1]],
                        "pattern": "conv_bn",
                        "description": "Conv + BatchNorm fusion (standard ONNX optimization)",
                    }
                )
            elif type_i == "linear" and type_j in ("normalization",):
                candidates.append(
                    {
                        "layers": [names[i], names[i + 1]],
                        "pattern": "linear_norm",
                        "description": "Linear + LayerNorm fusion (TensorRT plugin)",
                    }
                )

        # For transformer blocks, suggest attention + MLP fusion
        for name in names:
            layer_type = self.contributions[name].get("layer_type", "")
            if layer_type == "transformer_block":
                candidates.append(
                    {
                        "layers": [name],
                        "pattern": "transformer_block",
                        "description": "Full transformer block fusion (FlashAttention + fused MLP)",
                    }
                )

        return candidates

    def tensorrt_config(
        self,
        metric_name: str = "gradient_norm",
        workspace_mb: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate TensorRT builder configuration hints.

        Returns:
            Dict with recommended TensorRT builder settings
        """
        quant_plan = self.quantization_plan(metric_name=metric_name, target="fp16")
        prunable = self.prunable_layers()
        fusion = self.fusion_candidates()

        has_int8 = any(v == "int8" for v in quant_plan.values())

        return {
            "precision_mode": "fp16" if not has_int8 else "int8",
            "workspace_size_mb": workspace_mb,
            "layer_precisions": quant_plan,
            "prunable_layers": [p["layer_name"] for p in prunable],
            "fusion_opportunities": len(fusion),
            "num_layers": len(self.layer_names),
            "num_fp32_layers": sum(1 for v in quant_plan.values() if v == "fp32"),
            "num_reduced_precision_layers": sum(1 for v in quant_plan.values() if v != "fp32"),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export all hints as JSON-serializable dict."""
        return {
            "quantization_plan": self.quantization_plan(),
            "prunable_layers": self.prunable_layers(),
            "fusion_candidates": self.fusion_candidates(),
            "tensorrt_config": self.tensorrt_config(),
        }

    def _rank_by_metric(self, metric_name: str, ascending: bool = False) -> List[Tuple[str, float]]:
        """Rank layers by a metric."""
        ranked = []
        for name, metrics in self.contributions.items():
            val = metrics.get(metric_name)
            if val is not None:
                ranked.append((name, val))
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked
