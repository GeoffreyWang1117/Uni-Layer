"""
Mutual Information metric for layer contribution.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from uni_layer.core.base_metric import LayerMetric


class MutualInformation(LayerMetric):
    """
    Compute Mutual Information between layer activations and targets.

    MI(X; Y) measures how much information the layer activations contain
    about the target labels/values. Higher MI indicates that the layer
    captures more task-relevant information.

    Args:
        num_batches: Number of batches to use
        task_type: 'classification' or 'regression'
        n_neighbors: Number of neighbors for MI estimation
    """

    def __init__(
        self,
        num_batches: int = 10,
        task_type: str = "classification",
        n_neighbors: int = 3,
        **kwargs
    ):
        super().__init__(
            name="mutual_information",
            category="information_theory",
            requires_gradient=False,
            requires_data=True,
            **kwargs
        )
        self.num_batches = num_batches
        self.task_type = task_type
        self.n_neighbors = n_neighbors

    def compute(
        self,
        model: nn.Module,
        layer: nn.Module,
        layer_name: str,
        layer_idx: int,
        data_loader: Optional[Any] = None,
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute mutual information for the layer.

        Returns:
            Dictionary with MI scores
        """
        activations = []
        targets_list = []

        # Collect activations and targets
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        handle = layer.register_forward_hook(hook_fn)

        try:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= self.num_batches:
                        break

                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        # No targets available
                        return {"mutual_information": 0.0}

                    inputs = inputs.to(device)
                    model(inputs)
                    targets_list.append(targets.cpu())

        finally:
            handle.remove()

        if not activations or not targets_list:
            return {"mutual_information": 0.0}

        # Concatenate
        X = torch.cat(activations, dim=0)
        y = torch.cat(targets_list, dim=0)

        # Flatten activations
        if X.dim() > 2:
            X = X.reshape(X.size(0), -1)

        X = X.numpy()
        y = y.numpy()

        # Subsample if too large
        if X.shape[0] > 1000:
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X = X[indices]
            y = y[indices]

        # Compute MI
        try:
            if self.task_type == "classification":
                mi = mutual_info_classif(X, y, n_neighbors=self.n_neighbors, random_state=42)
            else:
                mi = mutual_info_regression(X, y, n_neighbors=self.n_neighbors, random_state=42)

            # Average MI across features
            mi_score = float(np.mean(mi))

            return {
                "mutual_information": mi_score,
                "mi_max": float(np.max(mi)),
                "mi_std": float(np.std(mi)),
            }

        except Exception as e:
            return {"mutual_information": 0.0}
