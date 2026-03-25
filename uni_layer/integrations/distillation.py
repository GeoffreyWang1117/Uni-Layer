"""
Integration bridge for layer-guided knowledge distillation.

This bridge helps select which intermediate layers to align during
knowledge distillation, using Uni-Layer's contribution analysis
instead of ad-hoc layer selection.

Works with any distillation framework or custom training loop.

Usage:
    >>> from uni_layer import LayerAnalyzer
    >>> from uni_layer.metrics import CKA, GradientNorm, EffectiveRank
    >>> from uni_layer.integrations import DistillationBridge
    >>>
    >>> # Analyze teacher model
    >>> analyzer = LayerAnalyzer(teacher_model, task_type='classification')
    >>> contributions = analyzer.compute_metrics(
    ...     metrics=[CKA(), GradientNorm(), EffectiveRank()],
    ...     data_loader=train_loader
    ... )
    >>>
    >>> # Get distillation recommendations
    >>> bridge = DistillationBridge(teacher_model, student_model, contributions)
    >>> layer_pairs = bridge.recommend_layer_pairs(top_k=4)
    >>> layer_weights = bridge.recommend_layer_weights(metric_name='cka_score')
    >>>
    >>> # Use in custom distillation loop
    >>> for teacher_layer, student_layer, weight in zip(...):
    ...     loss += weight * F.mse_loss(student_acts[student_layer],
    ...                                  teacher_acts[teacher_layer])
"""

from typing import Dict, List, Optional, Tuple
import torch.nn as nn
import numpy as np


class DistillationBridge:
    """
    Recommends layer pairing and weighting strategies for knowledge distillation.

    Key capabilities:
    1. Recommend which teacher layers to distill (based on contribution)
    2. Pair teacher layers with corresponding student layers (by relative depth)
    3. Assign per-layer distillation weights (important layers get higher weight)
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        contributions: Dict[str, Dict[str, float]],
    ):
        """
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            contributions: Output from LayerAnalyzer.compute_metrics() on teacher
        """
        self.teacher = teacher_model
        self.student = student_model
        self.contributions = contributions

        self._teacher_layers = list(teacher_model.named_modules())
        self._student_layers = list(student_model.named_modules())

    def recommend_distillation_layers(
        self,
        metric_name: str = "gradient_norm",
        top_k: int = 4,
    ) -> List[str]:
        """
        Recommend which teacher layers are most worth distilling.

        Layers with high contribution carry the most task-relevant
        information and are the best candidates for intermediate
        layer distillation.

        Args:
            metric_name: Metric to rank layers by
            top_k: Number of layers to recommend

        Returns:
            List of teacher layer names, sorted by importance
        """
        scores = {}
        for name, metrics in self.contributions.items():
            if metric_name in metrics and metrics[metric_name] is not None:
                scores[name] = float(metrics[metric_name])

        sorted_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_layers[:top_k]]

    def recommend_layer_pairs(
        self,
        metric_name: str = "gradient_norm",
        top_k: int = 4,
    ) -> List[Tuple[str, str]]:
        """
        Recommend teacher-student layer pairs for intermediate distillation.

        Pairs are created by matching teacher layers (selected by importance)
        to student layers at the same relative depth.

        Args:
            metric_name: Metric for teacher layer selection
            top_k: Number of layer pairs

        Returns:
            List of (teacher_layer_name, student_layer_name) tuples
        """
        teacher_layers = self.recommend_distillation_layers(metric_name, top_k)
        teacher_depth = len(self._teacher_layers)
        student_depth = len(self._student_layers)

        pairs = []
        for teacher_name in teacher_layers:
            # Find teacher layer index
            teacher_idx = None
            for idx, (name, _) in enumerate(self._teacher_layers):
                if name == teacher_name:
                    teacher_idx = idx
                    break

            if teacher_idx is None:
                continue

            # Map to student by relative position
            relative_pos = teacher_idx / max(teacher_depth, 1)
            student_idx = int(relative_pos * student_depth)
            student_idx = min(student_idx, len(self._student_layers) - 1)

            student_name = self._student_layers[student_idx][0]
            pairs.append((teacher_name, student_name))

        return pairs

    def recommend_layer_weights(
        self,
        metric_name: str = "gradient_norm",
        target_layers: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Compute per-layer distillation weights based on contribution.

        Important layers receive higher weight in the distillation loss.
        This focuses the student's capacity on learning the most
        task-relevant representations.

        Args:
            metric_name: Metric for weight computation
            target_layers: Layers to compute weights for (default: all)
            normalize: Whether weights should sum to 1.0

        Returns:
            Dict mapping layer names to distillation weights
        """
        scores = {}
        for name, metrics in self.contributions.items():
            if target_layers and name not in target_layers:
                continue
            if metric_name in metrics and metrics[metric_name] is not None:
                scores[name] = float(metrics[metric_name])

        if not scores:
            return {}

        if normalize:
            total = sum(scores.values())
            if total > 1e-8:
                scores = {k: v / total for k, v in scores.items()}

        return scores

    def summary(self, metric_name: str = "gradient_norm", top_k: int = 4) -> str:
        """Print a summary of distillation recommendations."""
        pairs = self.recommend_layer_pairs(metric_name, top_k)
        weights = self.recommend_layer_weights(metric_name)

        lines = [f"Uni-Layer -> Distillation Bridge (metric: {metric_name})"]
        lines.append(f"{'Teacher Layer':<35} {'Student Layer':<35} {'Weight':>8}")
        lines.append("-" * 80)

        for teacher, student in pairs:
            w = weights.get(teacher, 0.0)
            lines.append(f"{teacher:<35} {student:<35} {w:>8.4f}")

        return "\n".join(lines)
