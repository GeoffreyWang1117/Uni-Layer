"""
DropLayer metric for measuring layer importance through ablation.

Optimized version:
- Caches baseline evaluation results so they can be reused across layers
- Accepts pre-computed baseline via kwargs to avoid redundant forward passes
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import numpy as np

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.model_adapter import extract_logits, model_forward


class DropLayerRobustness(LayerMetric):
    """
    Measure layer importance by dropping/zeroing it out.

    This metric evaluates how much the model's performance degrades when
    a layer is removed or its outputs are zeroed. Larger performance drops
    indicate more important layers.

    Optimization: The baseline (model without any layer dropped) only needs
    to be computed once across all layers. Pass _baseline_results via kwargs
    to reuse it, or the metric will compute it automatically and cache it
    as a class attribute for subsequent calls with the same model.

    Args:
        num_batches: Number of batches to evaluate
        metric: Metric to measure ('loss', 'accuracy')
        drop_type: How to drop the layer ('zero', 'identity')
    """

    # Class-level baseline cache keyed by model id
    _baseline_cache: Dict[int, Dict] = {}

    def __init__(
        self,
        num_batches: int = 10,
        metric: str = "loss",
        drop_type: str = "zero",
        **kwargs
    ):
        super().__init__(
            name="droplayer_robustness",
            category="robustness",
            requires_gradient=False,
            requires_data=True,
            **kwargs
        )
        self.num_batches = num_batches
        self.metric = metric
        self.drop_type = drop_type

    def _evaluate_baseline(
        self,
        model: nn.Module,
        data_loader: Any,
        device: str,
        criterion: nn.Module,
    ) -> Dict[str, List[float]]:
        """Evaluate model without any layer dropped (baseline)."""
        model_id = id(model)

        # Check class-level cache
        if model_id in DropLayerRobustness._baseline_cache:
            return DropLayerRobustness._baseline_cache[model_id]

        baseline_losses = []
        baseline_accs = []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.num_batches:
                    break

                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    continue

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model_forward(model, inputs)
                logits = extract_logits(outputs)
                loss = criterion(logits, targets)
                baseline_losses.append(loss.item())

                if self.metric == "accuracy":
                    preds = logits.argmax(dim=1)
                    acc = (preds == targets).float().mean().item()
                    baseline_accs.append(acc)

        result = {"losses": baseline_losses, "accs": baseline_accs}
        DropLayerRobustness._baseline_cache[model_id] = result
        return result

    @classmethod
    def clear_baseline_cache(cls):
        """Clear cached baseline results (call when model changes)."""
        cls._baseline_cache.clear()

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
        _baseline_results: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute DropLayer robustness metric.

        The baseline evaluation is computed once and cached for all subsequent
        layers, cutting the number of forward passes nearly in half.

        Returns:
            Dictionary with performance drop metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Get or compute baseline (shared across all layers)
        if _baseline_results is not None:
            baseline = _baseline_results
        else:
            baseline = self._evaluate_baseline(model, data_loader, device, criterion)

        baseline_losses = baseline["losses"]
        baseline_accs = baseline["accs"]

        if not baseline_losses:
            return {
                "droplayer_loss_increase": 0.0,
                "droplayer_loss_ratio": 1.0,
            }

        # Evaluate with layer dropped
        dropped_losses = []
        dropped_accs = []

        def drop_hook(module, input, output):
            if self.drop_type == "zero":
                return torch.zeros_like(output)
            elif self.drop_type == "identity":
                if isinstance(input, tuple):
                    return input[0]
                return input

        handle = layer.register_forward_hook(drop_hook)

        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= self.num_batches:
                        break

                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        continue

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    try:
                        outputs = model_forward(model, inputs)
                        logits = extract_logits(outputs)
                        loss = criterion(logits, targets)
                        dropped_losses.append(loss.item())

                        if self.metric == "accuracy":
                            preds = logits.argmax(dim=1)
                            acc = (preds == targets).float().mean().item()
                            dropped_accs.append(acc)

                    except Exception:
                        dropped_losses.append(float('inf'))
                        if self.metric == "accuracy":
                            dropped_accs.append(0.0)

        finally:
            handle.remove()

        # Compute performance drop
        if baseline_losses and dropped_losses:
            baseline_loss = np.mean(baseline_losses)
            dropped_loss = np.mean(dropped_losses)
            loss_increase = dropped_loss - baseline_loss

            result = {
                "droplayer_loss_increase": float(loss_increase),
                "droplayer_loss_ratio": float(dropped_loss / baseline_loss) if baseline_loss > 0 else 1.0,
            }

            if self.metric == "accuracy" and baseline_accs and dropped_accs:
                baseline_acc = np.mean(baseline_accs)
                dropped_acc = np.mean(dropped_accs)
                acc_decrease = baseline_acc - dropped_acc

                result.update({
                    "droplayer_acc_decrease": float(acc_decrease),
                    "droplayer_acc_ratio": float(dropped_acc / baseline_acc) if baseline_acc > 0 else 0.0,
                })

            return result
        else:
            return {
                "droplayer_loss_increase": 0.0,
                "droplayer_loss_ratio": 1.0,
            }
