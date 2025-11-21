"""
DropLayer metric for measuring layer importance through ablation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
from copy import deepcopy

from uni_layer.core.base_metric import LayerMetric


class DropLayerRobustness(LayerMetric):
    """
    Measure layer importance by dropping/zeroing it out.

    This metric evaluates how much the model's performance degrades when
    a layer is removed or its outputs are zeroed. Larger performance drops
    indicate more important layers.

    Args:
        num_batches: Number of batches to evaluate
        metric: Metric to measure ('loss', 'accuracy')
        drop_type: How to drop the layer ('zero', 'identity')
    """

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

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute DropLayer robustness metric.

        Returns:
            Dictionary with performance drop metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Baseline performance (without dropping)
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

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                baseline_losses.append(loss.item())

                if self.metric == "accuracy":
                    preds = outputs.argmax(dim=1)
                    acc = (preds == targets).float().mean().item()
                    baseline_accs.append(acc)

        # Performance with layer dropped
        dropped_losses = []
        dropped_accs = []

        # Register hook to zero out layer output
        def drop_hook(module, input, output):
            if self.drop_type == "zero":
                return torch.zeros_like(output)
            elif self.drop_type == "identity":
                # Try to pass input through (identity)
                if isinstance(input, tuple):
                    return input[0]
                return input

        handle = layer.register_forward_hook(drop_hook)

        try:
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
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        dropped_losses.append(loss.item())

                        if self.metric == "accuracy":
                            preds = outputs.argmax(dim=1)
                            acc = (preds == targets).float().mean().item()
                            dropped_accs.append(acc)

                    except Exception:
                        # If dropping causes error, layer is critical
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
