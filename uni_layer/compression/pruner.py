"""
Automated pruning module based on layer contribution analysis.

Supports structured and unstructured pruning with differential strategies
based on layer importance metrics.
"""

from typing import Dict, List, Optional, Union
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy


class PruningStrategy(Enum):
    """Pruning strategies based on layer contributions"""
    MAGNITUDE = "magnitude"  # Standard magnitude-based pruning
    GRADIENT_NORM = "gradient_norm"  # Based on gradient norm metric
    FISHER = "fisher"  # Based on Fisher information
    HYBRID = "hybrid"  # Combination of multiple metrics


class LayerPruner:
    """
    Automated layer pruning based on contribution analysis.

    Mathematical Foundation:
    ========================

    For a neural network parameter θ, we define importance score I(θ) as:

    I(θ) = |θ| · |∂L/∂θ|  (for magnitude-based)
    I(θ) = ||∂L/∂θ||_2    (for gradient-based)
    I(θ) = E[(∂L/∂θ)²]    (for Fisher-based)

    Structured pruning removes entire units (neurons, channels, heads):
    θ_pruned = {θ_i : I(θ_i) > τ_i}

    where τ_i is a layer-specific threshold determined by contribution analysis.

    Differential Pruning:
    --------------------
    Layers with low contribution → high pruning ratio
    Layers with high contribution → low pruning ratio

    Pruning ratio for layer ℓ:
    r_ℓ = r_max · (1 - normalized_contribution_ℓ)

    Args:
        model: PyTorch model to prune
        contributions: Layer contribution dictionary from LayerAnalyzer
        strategy: Pruning strategy to use
    """

    def __init__(
        self,
        model: nn.Module,
        contributions: Dict[str, Dict[str, float]],
        strategy: PruningStrategy = PruningStrategy.GRADIENT_NORM
    ):
        self.model = model
        self.contributions = contributions
        self.strategy = strategy
        self.pruned_model = None
        self.pruning_masks = {}

    def compute_layer_pruning_ratios(
        self,
        base_ratio: float = 0.3,
        max_ratio: float = 0.8,
        min_ratio: float = 0.1,
        metric_name: str = "gradient_norm"
    ) -> Dict[str, float]:
        """
        Compute differential pruning ratios for each layer.

        Args:
            base_ratio: Base pruning ratio (default: 0.3)
            max_ratio: Maximum pruning ratio for low-contribution layers
            min_ratio: Minimum pruning ratio for high-contribution layers
            metric_name: Metric to use for determining importance

        Returns:
            Dictionary mapping layer names to pruning ratios
        """
        # Extract metric values
        metric_values = {}
        for layer_name, metrics in self.contributions.items():
            if metric_name in metrics:
                metric_values[layer_name] = metrics[metric_name]

        if not metric_values:
            raise ValueError(f"Metric '{metric_name}' not found in contributions")

        # Normalize contributions to [0, 1]
        min_val = min(metric_values.values())
        max_val = max(metric_values.values())

        if max_val - min_val < 1e-8:
            # All layers have similar contribution
            return {name: base_ratio for name in metric_values.keys()}

        normalized = {
            name: (val - min_val) / (max_val - min_val)
            for name, val in metric_values.items()
        }

        # Compute differential pruning ratios
        # High contribution → low pruning ratio
        # Low contribution → high pruning ratio
        pruning_ratios = {}
        for name, norm_val in normalized.items():
            ratio = max_ratio - (max_ratio - min_ratio) * norm_val
            pruning_ratios[name] = ratio

        return pruning_ratios

    def prune_unstructured(
        self,
        pruning_ratios: Optional[Dict[str, float]] = None,
        global_pruning: bool = False
    ) -> nn.Module:
        """
        Apply unstructured (weight-level) pruning.

        Unstructured pruning removes individual weights based on magnitude:
        W_pruned = W ⊙ M, where M_ij = I(|W_ij| > threshold)

        Args:
            pruning_ratios: Layer-specific pruning ratios (if None, compute automatically)
            global_pruning: If True, use global threshold across all layers

        Returns:
            Pruned model
        """
        if pruning_ratios is None:
            pruning_ratios = self.compute_layer_pruning_ratios()

        # Create a copy of the model
        self.pruned_model = copy.deepcopy(self.model)

        # Collect parameters to prune
        parameters_to_prune = []

        for name, module in self.pruned_model.named_modules():
            if name in pruning_ratios and hasattr(module, 'weight'):
                parameters_to_prune.append((module, 'weight'))

        if global_pruning:
            # Global pruning: apply same threshold across all layers
            avg_ratio = sum(pruning_ratios.values()) / len(pruning_ratios)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=avg_ratio,
            )
        else:
            # Layer-wise pruning: different threshold per layer
            for name, module in self.pruned_model.named_modules():
                if name in pruning_ratios and hasattr(module, 'weight'):
                    prune.l1_unstructured(
                        module,
                        name='weight',
                        amount=pruning_ratios[name]
                    )

        return self.pruned_model

    def prune_structured(
        self,
        pruning_ratios: Optional[Dict[str, float]] = None,
        dim: int = 0
    ) -> nn.Module:
        """
        Apply structured pruning (removes entire neurons/channels).

        Structured pruning removes entire structural units:
        - dim=0: Remove output neurons/channels
        - dim=1: Remove input neurons/channels

        Channel importance: I_c = ||W_c||_2 (L2 norm of channel weights)

        Args:
            pruning_ratios: Layer-specific pruning ratios
            dim: Dimension along which to prune (0 or 1)

        Returns:
            Pruned model
        """
        if pruning_ratios is None:
            pruning_ratios = self.compute_layer_pruning_ratios()

        self.pruned_model = copy.deepcopy(self.model)

        for name, module in self.pruned_model.named_modules():
            if name in pruning_ratios and isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=pruning_ratios[name],
                    n=2,  # L2 norm
                    dim=dim
                )

        return self.pruned_model

    def prune_gradual(
        self,
        initial_ratio: float = 0.0,
        final_ratio: float = 0.5,
        num_steps: int = 10,
        structured: bool = False
    ) -> List[nn.Module]:
        """
        Gradual pruning with progressive sparsity increase.

        Implements gradual magnitude pruning (Zhu et al., 2017):
        s_t = s_f + (s_i - s_f)(1 - (t - t_0)/(n·Δt))³

        where:
        - s_t: sparsity at step t
        - s_i: initial sparsity
        - s_f: final sparsity
        - n: total pruning steps

        Args:
            initial_ratio: Starting pruning ratio
            final_ratio: Final pruning ratio
            num_steps: Number of pruning steps
            structured: Whether to use structured pruning

        Returns:
            List of models at each pruning step
        """
        models = []

        for step in range(num_steps + 1):
            # Cubic sparsity schedule
            progress = step / num_steps
            current_ratio = final_ratio + (initial_ratio - final_ratio) * (1 - progress) ** 3

            # Compute ratios for this step
            base_ratios = self.compute_layer_pruning_ratios(base_ratio=current_ratio)

            # Apply pruning
            if structured:
                model = self.prune_structured(base_ratios)
            else:
                model = self.prune_unstructured(base_ratios)

            models.append(model)

        return models

    def remove_pruning_masks(self) -> nn.Module:
        """
        Make pruning permanent by removing masks and zeroed weights.

        Converts parameterization back to standard weights:
        W = W_orig ⊙ mask → W (permanently zeros pruned weights)

        Returns:
            Model with permanent pruning (no reparameterization)
        """
        if self.pruned_model is None:
            raise ValueError("No pruned model available. Run pruning first.")

        for module in self.pruned_model.modules():
            if hasattr(module, 'weight') and prune.is_pruned(module):
                prune.remove(module, 'weight')

        return self.pruned_model

    def get_sparsity_stats(self) -> Dict[str, float]:
        """
        Compute sparsity statistics for pruned model.

        Returns:
            Dictionary with sparsity statistics
        """
        if self.pruned_model is None:
            raise ValueError("No pruned model available")

        total_params = 0
        zero_params = 0
        layer_sparsity = {}

        for name, param in self.pruned_model.named_parameters():
            if 'weight' in name:
                layer_total = param.numel()
                layer_zero = (param == 0).sum().item()

                total_params += layer_total
                zero_params += layer_zero
                layer_sparsity[name] = layer_zero / layer_total

        overall_sparsity = zero_params / total_params if total_params > 0 else 0.0

        return {
            'overall_sparsity': overall_sparsity,
            'total_params': total_params,
            'zero_params': zero_params,
            'layer_sparsity': layer_sparsity
        }

    def estimate_speedup(self) -> Dict[str, float]:
        """
        Estimate theoretical speedup from pruning.

        Theoretical speedup (unstructured): limited by sparse ops support
        Theoretical speedup (structured): ~1/(1-sparsity) for FLOPS

        Returns:
            Dictionary with speedup estimates
        """
        stats = self.get_sparsity_stats()
        overall_sparsity = stats['overall_sparsity']

        # Theoretical FLOPS reduction
        flops_reduction = overall_sparsity
        theoretical_speedup = 1.0 / (1.0 - overall_sparsity) if overall_sparsity < 1.0 else float('inf')

        # Practical speedup (conservative estimate)
        # Unstructured pruning: 1.0-1.2x (limited by hardware support)
        # Structured pruning: closer to theoretical
        practical_speedup = 1.0 + overall_sparsity * 0.8  # Conservative estimate

        return {
            'theoretical_speedup': theoretical_speedup,
            'practical_speedup': practical_speedup,
            'flops_reduction': flops_reduction,
            'memory_reduction': overall_sparsity
        }
