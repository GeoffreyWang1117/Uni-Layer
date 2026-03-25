"""
Parameter-Efficient Fine-Tuning (PEFT) optimization based on layer contributions.

Provides optimal adapter/LoRA placement and rank selection based on layer
importance analysis.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import math


@dataclass
class AdapterConfig:
    """Configuration for PEFT methods"""
    method: str = "lora"  # lora, adapter, prefix_tuning
    rank: int = 8  # Rank for low-rank adaptation
    alpha: float = 16.0  # Scaling factor for LoRA
    dropout: float = 0.1  # Dropout rate
    target_modules: List[str] = None  # Target module types
    adaptive_rank: bool = True  # Use adaptive rank per layer


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.

    Mathematical Foundation:
    ========================
    LoRA approximates weight updates with low-rank decomposition:

    W' = W + ΔW = W + BA

    where:
    - W ∈ ℝ^(d×k): frozen pre-trained weights
    - B ∈ ℝ^(d×r), A ∈ ℝ^(r×k): trainable low-rank matrices
    - r << min(d,k): rank (typically 1-64)

    Forward pass:
    h = W·x + (α/r)·B·A·x

    where α is a scaling factor (typically 2r).

    Parameter efficiency:
    # trainable params = r(d + k) << dk

    Example: d=k=4096, r=8 → 99.8% parameter reduction

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        alpha: Scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices: W = BA
        # Initialize A with random Gaussian, B with zeros (following original paper)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = (α/r)·B·A·x

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA output (..., out_features)
        """
        # Apply dropout to input
        x = self.dropout(x)

        # Low-rank projection: A·x
        lora_output = x @ self.lora_A.T  # (..., rank)

        # Project back: B·(A·x)
        lora_output = lora_output @ self.lora_B.T  # (..., out_features)

        # Apply scaling
        return lora_output * self.scaling


class AdapterLayer(nn.Module):
    """
    Adapter layer (Houlsby et al., 2019).

    Mathematical Foundation:
    ========================
    Adapter uses bottleneck architecture:

    h' = h + f(hW_down)W_up

    where:
    - h ∈ ℝ^d: input activation
    - W_down ∈ ℝ^(d×r): down-projection
    - W_up ∈ ℝ^(r×d): up-projection
    - f: non-linearity (typically ReLU or GELU)
    - r << d: bottleneck dimension

    Residual connection preserves pre-trained representations.

    Args:
        input_dim: Input dimension
        bottleneck_dim: Bottleneck dimension (adapter rank)
        dropout: Dropout probability
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights
        nn.init.normal_(self.down_project.weight, std=1e-3)
        nn.init.normal_(self.up_project.weight, std=1e-3)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output with adapter: x + adapter(x)
        """
        residual = x

        # Down-project
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Up-project
        x = self.up_project(x)

        # Residual connection
        return residual + x


class PEFTOptimizer:
    """
    PEFT optimization based on layer contribution analysis.

    Determines optimal:
    1. Which layers to augment with adapters/LoRA
    2. Rank/capacity for each layer
    3. Training strategy

    Selection Strategy:
    ===================

    Layer Selection:
    ---------------
    Prioritize layers with:
    - High gradient norms → actively learning
    - High Fisher information → parameter-sensitive
    - High effective rank → rich representations

    Rank Selection:
    --------------
    Adaptive rank per layer based on:
    r_ℓ = r_base · (contribution_ℓ / max_contribution)

    Intuition:
    - Important layers get higher rank (more capacity)
    - Less important layers get lower rank (parameter efficiency)

    Total parameters:
    N_params = Σ_ℓ r_ℓ(d_ℓ + k_ℓ)

    Args:
        model: PyTorch model
        contributions: Layer contribution dictionary
        config: PEFT configuration
    """

    def __init__(
        self,
        model: nn.Module,
        contributions: Dict[str, Dict[str, float]],
        config: AdapterConfig = None
    ):
        self.model = model
        self.contributions = contributions
        self.config = config or AdapterConfig()
        self.peft_layers = {}

    def select_layers(
        self,
        top_k: int = 10,
        metric_name: str = "gradient_norm",
        min_contribution: float = 0.1
    ) -> List[str]:
        """
        Select which layers to augment with PEFT.

        Args:
            top_k: Number of layers to select
            metric_name: Metric to use for selection
            min_contribution: Minimum contribution threshold

        Returns:
            List of selected layer names
        """
        # Extract metric values
        layer_scores = {}

        for layer_name, metrics in self.contributions.items():
            if metric_name in metrics:
                layer_scores[layer_name] = metrics[metric_name]

        # Filter by minimum contribution
        layer_scores = {
            name: score
            for name, score in layer_scores.items()
            if score >= min_contribution
        }

        # Sort and select top-k
        sorted_layers = sorted(
            layer_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected = [name for name, _ in sorted_layers[:top_k]]

        return selected

    def compute_adaptive_ranks(
        self,
        selected_layers: List[str],
        base_rank: int = 8,
        max_rank: int = 64,
        metric_name: str = "gradient_norm"
    ) -> Dict[str, int]:
        """
        Compute adaptive rank for each layer based on contribution.

        Rank allocation strategy:
        r_ℓ = r_base + (r_max - r_base) · (contribution_ℓ / max_contribution)

        Args:
            selected_layers: Layers to compute ranks for
            base_rank: Minimum rank
            max_rank: Maximum rank
            metric_name: Metric for rank computation

        Returns:
            Dictionary mapping layer names to ranks
        """
        if not self.config.adaptive_rank:
            return {name: self.config.rank for name in selected_layers}

        # Get contributions for selected layers
        contributions = {}
        for name in selected_layers:
            if name in self.contributions and metric_name in self.contributions[name]:
                contributions[name] = self.contributions[name][metric_name]

        if not contributions:
            return {name: base_rank for name in selected_layers}

        # Normalize contributions
        max_contrib = max(contributions.values())
        min_contrib = min(contributions.values())

        if max_contrib - min_contrib < 1e-8:
            return {name: base_rank for name in selected_layers}

        # Compute adaptive ranks
        ranks = {}
        for name, contrib in contributions.items():
            normalized = (contrib - min_contrib) / (max_contrib - min_contrib)
            rank = int(base_rank + (max_rank - base_rank) * normalized)
            # Ensure rank is a multiple of 8 for efficiency
            rank = (rank // 8) * 8
            rank = max(8, min(rank, max_rank))  # Clamp to valid range
            ranks[name] = rank

        return ranks

    def inject_lora(
        self,
        selected_layers: Optional[List[str]] = None,
        ranks: Optional[Dict[str, int]] = None
    ) -> nn.Module:
        """
        Inject LoRA layers into the model.

        Args:
            selected_layers: Layers to inject LoRA into (if None, auto-select)
            ranks: Rank for each layer (if None, auto-compute)

        Returns:
            Model with LoRA layers injected
        """
        if selected_layers is None:
            selected_layers = self.select_layers()

        if ranks is None:
            ranks = self.compute_adaptive_ranks(selected_layers)

        # Inject LoRA into selected layers
        for name, module in self.model.named_modules():
            if name in selected_layers and isinstance(module, nn.Linear):
                # Get rank for this layer
                rank = ranks.get(name, self.config.rank)

                # Create LoRA layer
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout
                )

                # Freeze original weights
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

                # Store LoRA layer
                self.peft_layers[name] = lora

                # Replace forward method to include LoRA
                original_forward = module.forward

                def new_forward(x, orig_forward=original_forward, lora_layer=lora):
                    # Original output (frozen)
                    out = orig_forward(x)
                    # Add LoRA output
                    out = out + lora_layer(x)
                    return out

                module.forward = new_forward

        return self.model

    def inject_adapters(
        self,
        selected_layers: Optional[List[str]] = None,
        bottleneck_dim: int = 64
    ) -> nn.Module:
        """
        Inject adapter layers into the model.

        Args:
            selected_layers: Layers to inject adapters into
            bottleneck_dim: Adapter bottleneck dimension

        Returns:
            Model with adapters injected
        """
        if selected_layers is None:
            selected_layers = self.select_layers()

        for name, module in self.model.named_modules():
            if name in selected_layers:
                # Infer input dimension
                if isinstance(module, nn.Linear):
                    input_dim = module.out_features
                elif isinstance(module, nn.Conv2d):
                    input_dim = module.out_channels
                else:
                    continue

                # Create adapter
                adapter = AdapterLayer(
                    input_dim=input_dim,
                    bottleneck_dim=bottleneck_dim,
                    dropout=self.config.dropout
                )

                # Store adapter
                self.peft_layers[name] = adapter

                # Modify forward to include adapter
                original_forward = module.forward

                def new_forward(x, orig_forward=original_forward, adapter_layer=adapter):
                    out = orig_forward(x)
                    out = adapter_layer(out)
                    return out

                module.forward = new_forward

        return self.model

    def get_trainable_parameters(self) -> int:
        """
        Count trainable parameters after PEFT injection.

        Returns:
            Number of trainable parameters
        """
        total = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total += param.numel()

        return total

    def get_parameter_efficiency(self) -> Dict[str, float]:
        """
        Compute parameter efficiency statistics.

        Returns:
            Dictionary with efficiency metrics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = self.get_trainable_parameters()

        efficiency = trainable_params / total_params if total_params > 0 else 0.0

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'efficiency': efficiency,
            'reduction_ratio': 1.0 / efficiency if efficiency > 0 else float('inf'),
            'num_peft_layers': len(self.peft_layers)
        }

    def get_peft_info(self) -> Dict:
        """
        Get information about PEFT configuration.

        Returns:
            Dictionary with PEFT details
        """
        ranks = {}
        for name, layer in self.peft_layers.items():
            if isinstance(layer, LoRALayer):
                ranks[name] = layer.rank
            elif isinstance(layer, AdapterLayer):
                ranks[name] = layer.down_project.out_features

        return {
            'method': self.config.method,
            'num_layers': len(self.peft_layers),
            'layer_names': list(self.peft_layers.keys()),
            'ranks': ranks,
            'avg_rank': sum(ranks.values()) / len(ranks) if ranks else 0,
            'parameter_efficiency': self.get_parameter_efficiency()
        }
