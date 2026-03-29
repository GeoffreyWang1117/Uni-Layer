"""
Main LayerAnalyzer class for computing and analyzing layer contributions.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from uni_layer.core.base_metric import LayerMetric
from uni_layer.core.cache import ActivationCache, BatchCache, GradientCache
from uni_layer.utils.layer_utils import get_model_layers, identify_layer_type


class LayerAnalyzer:
    """
    Main analyzer class for computing layer contribution metrics.

    This class provides a unified interface for:
    - Computing multiple layer contribution metrics
    - Analyzing layer importance across different architectures
    - Providing layer-level recommendations for downstream optimization

    Use with integration bridges for actionable optimization:
    - uni_layer.integrations.TorchPruningBridge (pruning)
    - uni_layer.integrations.HuggingFacePEFTBridge (LoRA/Adapters)
    - uni_layer.integrations.DistillationBridge (distillation)

    Example:
        >>> analyzer = LayerAnalyzer(model, task_type='classification')
        >>> contributions = analyzer.compute_metrics(
        ...     metrics=[GradientNorm(), CKA()],
        ...     data_loader=train_loader
        ... )
        >>> rankings = analyzer.rank_layers(contributions, 'gradient_norm')
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

        print(f"Initialized LayerAnalyzer with {len(self.layers)} layers")
        print(f"  Device: {self.device}")
        print(f"  Task Type: {task_type}")

    # Preset metric configurations
    PRESETS = {
        "llm_fast": [
            "BlockInfluence",
            "EffectiveRank",
            "CKA",
            "ActivationEntropy",
            "AttentionFlow",
        ],
        "llm_full": [
            "BlockInfluence",
            "EffectiveRank",
            "CKA",
            "ActivationEntropy",
            "AttentionFlow",
            "GradientNorm",
            "FisherInformation",
        ],
        "full": [
            "GradientNorm",
            "HessianTrace",
            "FisherInformation",
            "CKA",
            "EffectiveRank",
            "NTKTrace",
            "ActivationEntropy",
            "MutualInformation",
            "JacobianRank",
            "BlockInfluence",
            "DropLayerRobustness",
            "ResidualDropLayer",
            "LaplacePosterior",
            "AttentionFlow",
        ],
        "quick": ["GradientNorm", "BlockInfluence", "EffectiveRank"],
    }

    def compute_metrics(
        self,
        metrics: Optional[List[LayerMetric]] = None,
        data_loader: Optional[Any] = None,
        layer_names: Optional[List[str]] = None,
        verbose: bool = True,
        use_cache: bool = True,
        num_batches: int = 10,
        preset: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute multiple layer contribution metrics.

        When use_cache=True (default), activations and gradients are captured
        once and shared across all metrics that need them, dramatically reducing
        the number of forward/backward passes.

        Args:
            metrics: List of LayerMetric instances to compute.
                     Ignored if preset is specified.
            data_loader: DataLoader for data-dependent metrics
            layer_names: Specific layers to analyze (default: all layers)
            verbose: Whether to show progress bars
            use_cache: Whether to cache activations/gradients (default: True)
            num_batches: Number of batches for cache capture
            preset: Use a named preset instead of manual metrics list.
                    Options: "llm_fast", "llm_full", "full", "quick"
            **kwargs: Additional arguments passed to metrics

        Returns:
            Dictionary mapping layer names to metric values
            Format: {layer_name: {metric_name: value}}
        """
        if preset is not None:
            metrics = self._resolve_preset(preset, num_batches)
        elif metrics is None:
            raise ValueError("Either metrics or preset must be provided")
        if layer_names is None:
            layer_names = list(self.layers.keys())

        results = OrderedDict()

        # Initialize results structure
        for layer_name in layer_names:
            results[layer_name] = {
                "layer_idx": list(self.layers.keys()).index(layer_name),
                "layer_type": self.layer_types[layer_name],
            }

        if use_cache and data_loader is not None:
            results = self._compute_with_cache(
                metrics, data_loader, layer_names, results, verbose, num_batches, **kwargs
            )
        else:
            results = self._compute_without_cache(
                metrics, data_loader, layer_names, results, verbose, **kwargs
            )

        return results

    def _compute_with_cache(
        self,
        metrics: List[LayerMetric],
        data_loader: Any,
        layer_names: List[str],
        results: OrderedDict,
        verbose: bool,
        num_batches: int,
        **kwargs,
    ) -> OrderedDict:
        """Compute metrics using shared activation/gradient caches."""

        # Cache data batches so we don't re-iterate the DataLoader
        batch_cache = BatchCache(data_loader, num_batches=num_batches, device=self.device)

        # Separate metrics by type
        needs_activation = [m for m in metrics if not m.requires_gradient]
        needs_gradient = [m for m in metrics if m.requires_gradient]

        # Build layer subset for caching
        layers_subset = {n: self.layers[n] for n in layer_names if n in self.layers}

        # Phase 1: Activation-based metrics (one forward pass for all layers)
        if needs_activation:
            if verbose:
                print(f"  Caching activations (1 forward pass for {len(layers_subset)} layers)...")

            act_cache = ActivationCache(self.model, layers_subset)
            act_cache.capture(batch_cache, num_batches=num_batches, device=self.device)

            for metric in tqdm(needs_activation, desc="Activation metrics", disable=not verbose):
                for layer_name in layer_names:
                    layer = self.layers.get(layer_name)
                    if layer is None:
                        continue
                    layer_idx = list(self.layers.keys()).index(layer_name)

                    try:
                        # Pass cached activations via kwargs
                        metric_values = metric.compute(
                            model=self.model,
                            layer=layer,
                            layer_name=layer_name,
                            layer_idx=layer_idx,
                            data_loader=batch_cache,
                            device=self.device,
                            criterion=self.criterion,
                            _cached_activations=act_cache.get(layer_name),
                            **kwargs,
                        )
                        results[layer_name].update(metric_values)
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: {metric.name} for {layer_name}: {e}")
                        results[layer_name][metric.name] = None

            del act_cache

        # Phase 2: Gradient-based metrics (must do own backward passes)
        if needs_gradient:
            if verbose:
                print(f"  Computing {len(needs_gradient)} gradient-based metrics...")

            for metric in tqdm(needs_gradient, desc="Gradient metrics", disable=not verbose):
                # Set model to train mode for gradient computation
                self.model.train()

                for layer_name in tqdm(
                    layer_names, desc=f"  {metric.name}", disable=not verbose, leave=False
                ):
                    layer = self.layers.get(layer_name)
                    if layer is None:
                        continue
                    layer_idx = list(self.layers.keys()).index(layer_name)

                    try:
                        metric_values = metric.compute(
                            model=self.model,
                            layer=layer,
                            layer_name=layer_name,
                            layer_idx=layer_idx,
                            data_loader=batch_cache,
                            device=self.device,
                            criterion=self.criterion,
                            **kwargs,
                        )
                        results[layer_name].update(metric_values)
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: {metric.name} for {layer_name}: {e}")
                        results[layer_name][metric.name] = None

        return results

    def _compute_without_cache(
        self,
        metrics: List[LayerMetric],
        data_loader: Any,
        layer_names: List[str],
        results: OrderedDict,
        verbose: bool,
        **kwargs,
    ) -> OrderedDict:
        """Original compute path (no caching)."""
        for metric in tqdm(metrics, desc="Computing metrics", disable=not verbose):
            if metric.requires_data and data_loader is None:
                if verbose:
                    print(f"  Skipping {metric.name}: requires data_loader")
                continue

            if metric.requires_gradient:
                self.model.train()
            else:
                self.model.eval()

            for layer_name in tqdm(
                layer_names, desc=f"  {metric.name}", disable=not verbose, leave=False
            ):
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
                        **kwargs,
                    )
                    results[layer_name].update(metric_values)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: {metric.name} for {layer_name}: {e}")
                    results[layer_name][metric.name] = None

        return results

    def rank_layers(
        self, contributions: Dict[str, Dict[str, float]], metric_name: str, ascending: bool = False
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
        ascending: bool = False,
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
        prune_ratio: float = 0.3,
    ) -> Dict[str, float]:
        """
        Recommend layer-wise pruning ratios based on contributions.

        For production pruning workflows, use this with
        uni_layer.integrations.TorchPruningBridge instead.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to base pruning on
            prune_ratio: Overall pruning ratio (0-1)

        Returns:
            Dictionary mapping layer names to recommended pruning ratios
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
        top_k: int = 6,
    ) -> List[str]:
        """
        Recommend layers for knowledge distillation based on contributions.

        For production distillation workflows, use this with
        uni_layer.integrations.DistillationBridge instead.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to base selection on
            top_k: Number of layers to select

        Returns:
            List of recommended layer names for distillation
        """
        return self.get_top_k_layers(contributions, metric_name, k=top_k)

    def get_peft_insertion_points(
        self,
        contributions: Dict[str, Dict[str, float]],
        metric_name: str = "gradient_norm",
        num_adapters: int = 4,
    ) -> List[str]:
        """
        Recommend insertion points for parameter-efficient fine-tuning adapters.

        For production PEFT workflows, use this with
        uni_layer.integrations.HuggingFacePEFTBridge instead.

        Args:
            contributions: Output from compute_metrics()
            metric_name: Metric to base selection on
            num_adapters: Number of adapter insertion points

        Returns:
            List of recommended layer names for adapter insertion
        """
        return self.get_top_k_layers(contributions, metric_name, k=num_adapters)

    def aggregate_by_depth(
        self, contributions: Dict[str, Dict[str, float]], metric_name: str, num_bins: int = 5
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
        self, contributions: Dict[str, Dict[str, float]], metric_name: str
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

    def _resolve_preset(self, preset: str, num_batches: int) -> List[LayerMetric]:
        """Resolve a preset name to a list of metric instances."""
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(self.PRESETS.keys())}")

        import uni_layer.metrics as m

        metric_map = {
            "GradientNorm": m.GradientNorm,
            "HessianTrace": m.HessianTrace,
            "FisherInformation": m.FisherInformation,
            "WandaImportance": m.WandaImportance,
            "IGSensitivity": m.IGSensitivity,
            "CKA": m.CKA,
            "EffectiveRank": m.EffectiveRank,
            "NTKTrace": m.NTKTrace,
            "ActivationEntropy": m.ActivationEntropy,
            "MutualInformation": m.MutualInformation,
            "JacobianRank": m.JacobianRank,
            "BlockInfluence": m.BlockInfluence,
            "DropLayerRobustness": m.DropLayerRobustness,
            "ResidualDropLayer": m.ResidualDropLayer,
            "LaplacePosterior": m.LaplacePosterior,
            "EfficiencyProfiler": m.EfficiencyProfiler,
            "WeightDistribution": m.WeightDistribution,
            "IntrinsicDimensionality": m.IntrinsicDimensionality,
            "QuantizationSensitivity": m.QuantizationSensitivity,
            "AdversarialSensitivity": m.AdversarialSensitivity,
            "ActivationAnomalyScore": m.ActivationAnomalyScore,
            "MembershipInferenceRisk": m.MembershipInferenceRisk,
            "AttentionPathTrace": m.AttentionPathTrace,
            "AttentionFlow": m.AttentionFlow,
            "MoERouterAnalysis": m.MoERouterAnalysis,
            "DiffusionTimestepAnalysis": m.DiffusionTimestepAnalysis,
        }

        instances = []
        for name in self.PRESETS[preset]:
            cls = metric_map.get(name)
            if cls:
                instances.append(cls(num_batches=num_batches))
        return instances

    def compute_cka_matrix(
        self,
        data_loader: Optional[Any] = None,
        num_batches: int = 10,
        verbose: bool = True,
        cka_batch_size: int = 256,
    ):
        """
        Compute the layer-to-layer CKA similarity matrix.

        Returns an N×N matrix where entry (i, j) is the CKA score between
        layer i and layer j. Use this to identify redundant layers and
        representation diversity across depth.

        Args:
            data_loader: DataLoader providing input data
            num_batches: Number of batches for activation capture
            verbose: Whether to print progress
            cka_batch_size: Batch size for minibatch CKA (memory control)

        Returns:
            Tuple of (similarity_matrix, layer_names)

        Example:
            >>> matrix, names = analyzer.compute_cka_matrix(train_loader)
            >>> # matrix[i][j] = CKA similarity between layer i and j
        """
        from uni_layer.core.cka_similarity import CKASimilarity

        cka_sim = CKASimilarity(
            model=self.model,
            layers=self.layers,
            cka_batch_size=cka_batch_size,
            device=self.device,
        )
        return cka_sim.compute(
            data_loader=data_loader,
            num_batches=num_batches,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        return (
            f"LayerAnalyzer(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  num_layers={len(self.layers)},\n"
            f"  task_type='{self.task_type}',\n"
            f"  device='{self.device}'\n"
            f")"
        )
