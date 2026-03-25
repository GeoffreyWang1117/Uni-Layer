"""
Integration bridge between Uni-Layer contribution analysis and HuggingFace PEFT.

HuggingFace PEFT (https://github.com/huggingface/peft) provides production-ready
LoRA, AdaLoRA, Adapter, and other parameter-efficient fine-tuning methods.

Uni-Layer enhances PEFT by answering two critical questions:
1. Which layers should receive adapters? (layer selection)
2. What rank should each adapter have? (adaptive rank allocation)

Usage:
    >>> from uni_layer import LayerAnalyzer
    >>> from uni_layer.metrics import GradientNorm, FisherInformation
    >>> from uni_layer.integrations import HuggingFacePEFTBridge
    >>>
    >>> # Step 1: Analyze layer contributions
    >>> analyzer = LayerAnalyzer(model, task_type='classification')
    >>> contributions = analyzer.compute_metrics(
    ...     metrics=[GradientNorm(), FisherInformation()],
    ...     data_loader=train_loader
    ... )
    >>>
    >>> # Step 2: Get PEFT recommendations
    >>> bridge = HuggingFacePEFTBridge(model, contributions)
    >>> target_modules = bridge.recommend_target_modules(
    ...     metric_name='gradient_norm', top_k=8
    ... )
    >>> rank_mapping = bridge.recommend_adaptive_ranks(
    ...     metric_name='gradient_norm', base_rank=8, max_rank=64
    ... )
    >>>
    >>> # Step 3: Use with HuggingFace PEFT
    >>> from peft import LoraConfig, get_peft_model
    >>> lora_config = LoraConfig(
    ...     target_modules=target_modules,
    ...     r=8,  # or use rank_mapping with AdaLoRA
    ...     lora_alpha=16,
    ... )
    >>> peft_model = get_peft_model(model, lora_config)

Requires: pip install peft
"""

from typing import Dict, List, Optional
import torch.nn as nn
import numpy as np


class HuggingFacePEFTBridge:
    """
    Converts Uni-Layer contribution analysis into recommendations
    for HuggingFace PEFT configurations.

    Key capabilities:
    1. Recommend which modules to target for LoRA/Adapter injection
    2. Compute adaptive rank allocation per layer
    3. Generate ready-to-use PEFT config parameters
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

    def recommend_target_modules(
        self,
        metric_name: str = "gradient_norm",
        top_k: Optional[int] = None,
        top_ratio: float = 0.5,
        module_types: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Recommend which modules to target for PEFT injection.

        Selects layers with highest contribution scores. By default targets
        Linear layers (the standard for LoRA), but can be customized.

        Args:
            metric_name: Metric to rank layers by
            top_k: Exact number of modules to select (overrides top_ratio)
            top_ratio: Fraction of eligible modules to select (default: 50%)
            module_types: Module type names to consider (default: ["Linear"])

        Returns:
            List of module names suitable for PEFT's target_modules parameter
        """
        module_types = module_types or ["Linear"]

        # Find eligible modules with contribution scores
        eligible = {}
        for name, module in self.model.named_modules():
            type_name = type(module).__name__
            if type_name not in module_types:
                continue

            # Match to contribution data (try exact match, then suffix match)
            score = self._find_score(name, metric_name)
            if score is not None:
                eligible[name] = score

        if not eligible:
            # Fallback: return common patterns for transformers
            return self._fallback_target_modules()

        # Select top layers
        sorted_modules = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        k = top_k if top_k is not None else max(1, int(len(sorted_modules) * top_ratio))
        selected = [name for name, _ in sorted_modules[:k]]

        return selected

    def recommend_adaptive_ranks(
        self,
        metric_name: str = "gradient_norm",
        base_rank: int = 8,
        max_rank: int = 64,
        target_modules: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Compute adaptive LoRA rank for each layer based on contribution.

        More important layers get higher rank (more expressive power),
        less important layers get lower rank (more parameter efficient).

        Formula: rank_l = base + (max - base) * normalized_contribution_l
        Ranks are rounded to multiples of 8 for hardware efficiency.

        Args:
            metric_name: Metric for rank allocation
            base_rank: Minimum rank for any layer
            max_rank: Maximum rank for any layer
            target_modules: Modules to compute ranks for (if None, use all)

        Returns:
            Dict mapping module names to recommended ranks
        """
        if target_modules is None:
            target_modules = self.recommend_target_modules(metric_name)

        # Collect scores
        scores = {}
        for name in target_modules:
            score = self._find_score(name, metric_name)
            if score is not None:
                scores[name] = score

        if not scores:
            return {name: base_rank for name in target_modules}

        # Normalize to [0, 1]
        min_s = min(scores.values())
        max_s = max(scores.values())
        if max_s - min_s < 1e-8:
            return {name: base_rank for name in target_modules}

        ranks = {}
        for name, score in scores.items():
            normalized = (score - min_s) / (max_s - min_s)
            rank = int(base_rank + (max_rank - base_rank) * normalized)
            rank = max(base_rank, (rank // 8) * 8)  # Round to multiple of 8
            ranks[name] = rank

        return ranks

    def recommend_lora_config_params(
        self,
        metric_name: str = "gradient_norm",
        top_k: Optional[int] = None,
        base_rank: int = 8,
    ) -> Dict:
        """
        Generate recommended parameters for peft.LoraConfig.

        Returns a dictionary that can be unpacked directly into LoraConfig:
            config = LoraConfig(**bridge.recommend_lora_config_params())

        Args:
            metric_name: Metric for layer selection
            top_k: Number of target modules
            base_rank: Base LoRA rank

        Returns:
            Dict with keys: target_modules, r, lora_alpha, task_type
        """
        target_modules = self.recommend_target_modules(
            metric_name=metric_name, top_k=top_k
        )

        return {
            "target_modules": target_modules,
            "r": base_rank,
            "lora_alpha": base_rank * 2,
            "lora_dropout": 0.1,
            "bias": "none",
        }

    def summary(self, metric_name: str = "gradient_norm") -> str:
        """Print a summary of PEFT recommendations."""
        targets = self.recommend_target_modules(metric_name)
        ranks = self.recommend_adaptive_ranks(metric_name, target_modules=targets)

        lines = [f"Uni-Layer -> HuggingFace PEFT Bridge (metric: {metric_name})"]
        lines.append(f"Selected {len(targets)} target modules:")
        lines.append(f"{'Module':<50} {'Rank':>6}")
        lines.append("-" * 58)

        for name in targets:
            rank = ranks.get(name, 8)
            lines.append(f"{name:<50} {rank:>6}")

        total_rank_params = sum(ranks.values()) * 2  # Rough estimate
        lines.append(f"\nEstimated LoRA parameter overhead: ~{total_rank_params}x hidden_dim")

        return "\n".join(lines)

    def _find_score(self, module_name: str, metric_name: str) -> Optional[float]:
        """Find contribution score for a module, with fuzzy matching."""
        # Exact match
        if module_name in self.contributions:
            metrics = self.contributions[module_name]
            if metric_name in metrics and metrics[metric_name] is not None:
                return float(metrics[metric_name])

        # Try parent layer match (e.g., "layer.0.attention" matches "layer.0")
        for contrib_name, metrics in self.contributions.items():
            if module_name.startswith(contrib_name) or contrib_name.startswith(module_name):
                if metric_name in metrics and metrics[metric_name] is not None:
                    return float(metrics[metric_name])

        return None

    def _fallback_target_modules(self) -> List[str]:
        """Return common target module patterns for transformer models."""
        patterns = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                short = name.split(".")[-1]
                if short in ("q_proj", "k_proj", "v_proj", "o_proj",
                             "query", "key", "value", "dense",
                             "fc1", "fc2", "out_proj"):
                    if short not in patterns:
                        patterns.append(short)
        return patterns if patterns else ["q_proj", "v_proj"]
