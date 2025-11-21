"""
Main LayerAnalyzer class for computing and analyzing layer contributions.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

from uni_layer.core.base_metric import LayerMetric
from uni_layer.utils.layer_utils import get_model_layers, identify_layer_type


class LayerAnalyzer:
    """
    Main analyzer class for computing layer contribution metrics.

    This class provides a unified interface for:
    - Computing multiple layer contribution metrics
    - Analyzing layer importance across different architectures
    - Generating insights for downstream tasks (pruning, distillation, PEFT)

    Example:
        >>> analyzer = LayerAnalyzer(model, task_type='classification')
        >>> contributions = analyzer.compute_metrics(
        ...     metrics=[GradientNorm(), CKA()],
        ...     data_loader=train_loader
        ... )
        >>> analyzer.visualize(contributions)
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "classification",
        device: Optional[str] = None,
        criterion: Optional[nn.Module] = None,
    ):
        """
        Initialize LayerAnalyzer.

        Args:
            model: PyTorch model to analyze
            task_type: Type of task ('classification', 'regression', 'generation', etc.)
            device: Device to run computations on (default: auto-detect)
            criterion: Loss function (default: based on task_type)
        """
        self.model = model
        self.task_type = task_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set default criterion based on task type
        if criterion is None:
            if task_type == "classification":
                self.criterion = nn.CrossEntropyLoss()
            elif task_type == "regression":
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        # Extract model layers
        self.layers = get_model_layers(model)
        self.layer_types = {name: identify_layer_type(layer) for name, layer in self.layers.items()}

        print(f"✓ Initialized LayerAnalyzer with {len(self.layers)} layers")
        print(f"  Device: {self.device}")
        print(f"  Task Type: {task_type}")

    def compute_metrics(
        self,
        metrics: List[LayerMetric],
        data_loader: Optional[Any] = None,
        layer_names: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute multiple layer contribution metrics.

        Args:
            metrics: List of LayerMetric instances to compute
            data_loader: DataLoader for data-dependent metrics
            layer_names: Specific layers to analyze (default: all layers)
            verbose: Whether to show progress bars
            **kwargs: Additional arguments passed to metrics

        Returns:
            Dictionary mapping layer names to metric values
            Format: {layer_name: {metric_name: value}}
        """
        if layer_names is None:
            layer_names = list(self.layers.keys())

        results = OrderedDict()

        # Initialize results structure
        for layer_name in layer_names:
            results[layer_name] = {
                "layer_idx": list(self.layers.keys()).index(layer_name),
                "layer_type": self.layer_types[layer_name],
            }

        # Compute each metric
        for metric in tqdm(metrics, desc="Computing metrics", disable=not verbose):
            if metric.requires_data and data_loader is None:
                print(f"⚠ Skipping {metric.name}: requires data_loader")
                continue

            # Set model to appropriate mode
            if metric.requires_gradient:
                self.model.train()
            else:
                self.model.eval()

            for layer_name in tqdm(layer_names, desc=f"  {metric.name}", disable=not verbose, leave=False):
                layer = self.layers[layer_name]
                layer_idx = list(self.layers.keys()).index(layer_name)

                try:
                    metric_values = metric.compute(
                        model=self.model,
                        layer=layer,
                        layer_name=layer_name,
                        layer_idx=layer_idx,
                        data_loader=data_loader,
                        device=self.device,
                        criterion=self.criterion,
                        **kwargs
                    )

                    # Merge metric values into results
                    results[layer_name].update(metric_values)

                except Exception as e:
                    if verbose:
                        print(f"⚠ Error computing {metric.name} for {layer_name}: {e}")
                    results[layer_name][metric.name] = None

        return results

    def rank_layers(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str,
        ascending: bool = False
    ) -> List[tuple]:
        """
        Rank layers by a specific metric.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Name of the metric to rank by
            ascending: If True, rank in ascending order (lower is better)

        Returns:
            List of (layer_name, metric_value) tuples, sorted
        """
        rankings = []
        for layer_name, metrics in contributions.items():
            if metric_name in metrics and metrics[metric_name] is not None:
                rankings.append((layer_name, metrics[metric_name]))

        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings

    def get_top_k_layers(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str,
        k: int = 5,
        ascending: bool = False
    ) -> List[str]:
        """
        Get top-k most important layers based on a metric.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Name of the metric to use
            k: Number of top layers to return
            ascending: If True, select lowest values instead

        Returns:
            List of layer names
        """
        rankings = self.rank_layers(contributions, metric_name, ascending)
        return [name for name, _ in rankings[:k]]

    def get_pruning_strategy(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str = "gradient_norm",
        prune_ratio: float = 0.3
    ) -> Dict[str, float]:
        """
        Generate layer-wise pruning strategy based on contributions.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to base pruning on
            prune_ratio: Overall pruning ratio (0-1)

        Returns:
            Dictionary mapping layer names to pruning ratios
        """
        rankings = self.rank_layers(contributions, metric_name, ascending=True)

        # Less important layers get higher pruning ratios
        strategy = {}
        num_layers = len(rankings)

        for i, (layer_name, _) in enumerate(rankings):
            # Linear strategy: least important gets 2x prune_ratio, most important gets 0
            layer_ratio = prune_ratio * 2 * (1 - i / num_layers)
            strategy[layer_name] = min(layer_ratio, 0.9)  # Cap at 90%

        return strategy

    def get_distillation_layers(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str = "gradient_norm",
        top_k: int = 6
    ) -> List[str]:
        """
        Select layers for knowledge distillation based on contributions.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to base selection on
            top_k: Number of layers to select

        Returns:
            List of layer names for distillation
        """
        return self.get_top_k_layers(contributions, metric_name, k=top_k)

    def get_peft_insertion_points(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str = "gradient_norm",
        num_adapters: int = 4
    ) -> List[str]:
        """
        Identify optimal insertion points for parameter-efficient fine-tuning adapters.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to base selection on
            num_adapters: Number of adapter insertion points

        Returns:
            List of layer names for adapter insertion
        """
        return self.get_top_k_layers(contributions, metric_name, k=num_adapters)

    def aggregate_by_depth(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str,
        num_bins: int = 5
    ) -> Dict[str, float]:
        """
        Aggregate layer contributions by depth (early, middle, late layers).

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to aggregate
            num_bins: Number of depth bins

        Returns:
            Dictionary mapping depth bins to aggregated values
        """
        num_layers = len(contributions)
        bin_size = num_layers // num_bins
        bins = {}

        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else num_layers
            bin_name = f"depth_{i+1}"

            values = []
            for j, (layer_name, metrics) in enumerate(contributions.items()):
                if start_idx <= j < end_idx and metric_name in metrics:
                    if metrics[metric_name] is not None:
                        values.append(metrics[metric_name])

            bins[bin_name] = np.mean(values) if values else 0.0

        return bins

    def get_summary_statistics(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str
    ) -> Dict[str, float]:
        """
        Get summary statistics for a metric across all layers.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to summarize

        Returns:
            Dictionary with mean, std, min, max, etc.
        """
        values = []
        for metrics in contributions.values():
            if metric_name in metrics and metrics[metric_name] is not None:
                values.append(metrics[metric_name])

        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
        }

    def __repr__(self) -> str:
        return (
            f"LayerAnalyzer(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  num_layers={len(self.layers)},\n"
            f"  task_type='{self.task_type}',\n"
            f"  device='{self.device}'\n"
            f")"
        )
