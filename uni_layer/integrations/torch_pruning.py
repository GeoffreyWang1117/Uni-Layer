"""
Integration bridge between Uni-Layer contribution analysis and Torch-Pruning.

Torch-Pruning (https://github.com/VainF/Torch-Pruning) provides structural pruning
via Dependency Graphs (DepGraph). Uni-Layer enhances it by replacing simple magnitude
heuristics with 30+ principled contribution metrics.

Usage:
    >>> from uni_layer import LayerAnalyzer
    >>> from uni_layer.metrics import GradientNorm, CKA, FisherInformation
    >>> from uni_layer.integrations import TorchPruningBridge
    >>>
    >>> # Step 1: Analyze layer contributions
    >>> analyzer = LayerAnalyzer(model, task_type='classification')
    >>> contributions = analyzer.compute_metrics(
    ...     metrics=[GradientNorm(), FisherInformation()],
    ...     data_loader=train_loader
    ... )
    >>>
    >>> # Step 2: Bridge to Torch-Pruning
    >>> bridge = TorchPruningBridge(model, contributions)
    >>> importance = bridge.as_importance_scores(metric_name='gradient_norm')
    >>> pruning_ratios = bridge.as_layer_pruning_ratios(
    ...     metric_name='gradient_norm', target_sparsity=0.5
    ... )
    >>>
    >>> # Step 3: Use with Torch-Pruning
    >>> import torch_pruning as tp
    >>> pruner = tp.pruner.MetaPruner(
    ...     model, example_inputs,
    ...     importance=tp.importance.MagnitudeImportance(),
    ...     pruning_ratio_dict=pruning_ratios,
    ... )

Requires: pip install torch-pruning
"""

from typing import Dict, List, Optional, Any
import torch.nn as nn
import numpy as np


class TorchPruningBridge:
    """
    Converts Uni-Layer contribution analysis into formats compatible
    with Torch-Pruning's API.

    This bridge provides three key capabilities:
    1. Convert metrics to importance scores (for custom importance criteria)
    2. Compute per-layer pruning ratios (for MetaPruner's pruning_ratio_dict)
    3. Identify layers to protect from pruning (high-contribution layers)
    """

    def __init__(
        self,
        model: nn.Module,
        contributions: Dict[str, Dict[str, float]],
    ):
        """
        Args:
            model: The PyTorch model being analyzed
            contributions: Output from LayerAnalyzer.compute_metrics()
        """
        self.model = model
        self.contributions = contributions
        self._module_name_map = {
            name: module for name, module in model.named_modules()
        }

    def as_importance_scores(
        self,
        metric_name: str = "gradient_norm",
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Convert contribution metrics to importance scores.

        Returns a dict mapping layer names to [0, 1] importance scores,
        suitable for use as custom importance criteria in Torch-Pruning.

        Args:
            metric_name: Which metric to use as importance signal
            normalize: Whether to normalize scores to [0, 1]

        Returns:
            Dict mapping layer names to importance scores
        """
        scores = {}
        for layer_name, metrics in self.contributions.items():
            if metric_name in metrics and metrics[metric_name] is not None:
                scores[layer_name] = float(metrics[metric_name])

        if not scores:
            raise ValueError(
                f"Metric '{metric_name}' not found in contributions. "
                f"Available: {self._available_metrics()}"
            )

        if normalize and scores:
            min_val = min(scores.values())
            max_val = max(scores.values())
            if max_val - min_val > 1e-8:
                scores = {
                    k: (v - min_val) / (max_val - min_val)
                    for k, v in scores.items()
                }

        return scores

    def as_layer_pruning_ratios(
        self,
        metric_name: str = "gradient_norm",
        target_sparsity: float = 0.5,
        max_ratio: float = 0.8,
        min_ratio: float = 0.0,
        protected_layers: Optional[List[str]] = None,
    ) -> Dict[nn.Module, float]:
        """
        Compute per-layer pruning ratios for Torch-Pruning's MetaPruner.

        Important layers (high contribution) get low pruning ratios;
        unimportant layers get high pruning ratios.

        Formula: ratio_l = max_ratio * (1 - normalized_contribution_l)

        Args:
            metric_name: Metric to drive pruning decisions
            target_sparsity: Target overall sparsity (0-1)
            max_ratio: Maximum pruning ratio for any layer
            min_ratio: Minimum pruning ratio for any layer
            protected_layers: Layer names to exclude from pruning

        Returns:
            Dict mapping nn.Module to pruning ratio (Torch-Pruning format)
        """
        protected = set(protected_layers or [])

        importance = self.as_importance_scores(metric_name, normalize=True)

        ratios = {}
        for layer_name, score in importance.items():
            if layer_name in protected:
                continue

            module = self._module_name_map.get(layer_name)
            if module is None:
                continue

            # Inverse importance: low contribution = high pruning ratio
            ratio = max_ratio * (1.0 - score)
            ratio = max(min_ratio, min(ratio, max_ratio))
            ratios[module] = ratio

        # Scale ratios to approximate target sparsity
        if ratios:
            current_mean = np.mean(list(ratios.values()))
            if current_mean > 1e-8:
                scale = target_sparsity / current_mean
                ratios = {
                    m: max(min_ratio, min(r * scale, max_ratio))
                    for m, r in ratios.items()
                }

        return ratios

    def get_protected_layers(
        self,
        metric_name: str = "gradient_norm",
        top_k: int = 3,
    ) -> List[str]:
        """
        Identify layers that should be protected from pruning.

        Returns the top-k most important layers based on contribution
        analysis. These layers should be excluded or given minimal
        pruning ratios.

        Args:
            metric_name: Metric to determine importance
            top_k: Number of layers to protect

        Returns:
            List of layer names to protect
        """
        scores = self.as_importance_scores(metric_name, normalize=True)
        sorted_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_layers[:top_k]]

    def summary(self, metric_name: str = "gradient_norm") -> str:
        """Print a summary of the bridge analysis."""
        scores = self.as_importance_scores(metric_name, normalize=True)
        sorted_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        lines = [f"Uni-Layer -> Torch-Pruning Bridge (metric: {metric_name})"]
        lines.append(f"{'Layer':<40} {'Importance':>10} {'Suggested Prune':>15}")
        lines.append("-" * 67)

        for name, score in sorted_layers:
            prune = f"{(1.0 - score) * 0.8:.1%}"
            lines.append(f"{name:<40} {score:>10.4f} {prune:>15}")

        return "\n".join(lines)

    def _available_metrics(self) -> List[str]:
        """List available metrics in contributions."""
        all_metrics = set()
        for metrics in self.contributions.values():
            all_metrics.update(
                k for k, v in metrics.items()
                if k not in ("layer_idx", "layer_type") and v is not None
            )
        return sorted(all_metrics)
