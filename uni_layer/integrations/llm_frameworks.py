"""
LLM Training Framework integration (Axolotl / LLaMA-Factory).

Converts Uni-Layer analysis results into configuration files for popular
LLM fine-tuning frameworks, enabling metrics-driven training decisions.

Supports:
- Axolotl: YAML config with LoRA target modules and ranks
- LLaMA-Factory: JSON config with adapter and freezing settings
"""

import json
from typing import Any, Dict, List, Optional

import torch.nn as nn
import yaml


class AxolotlConfigBridge:
    """
    Generate Axolotl training config from Uni-Layer analysis.

    Produces YAML configuration with:
    - LoRA target modules selected by layer importance
    - Adaptive rank allocation based on metric scores
    - Layer freezing recommendations

    Example:
        >>> bridge = AxolotlConfigBridge(model, contributions)
        >>> config = bridge.generate_config(base_rank=16)
        >>> bridge.save_yaml("axolotl_config.yml")
    """

    def __init__(
        self,
        model: nn.Module,
        contributions: Dict[str, Dict[str, Any]],
    ):
        self.model = model
        self.contributions = contributions

    def recommend_target_modules(
        self,
        metric_name: str = "gradient_norm",
        top_k: Optional[int] = None,
        top_ratio: float = 0.5,
    ) -> List[str]:
        """
        Recommend LoRA target modules based on importance.

        Returns module name patterns suitable for Axolotl's target_modules config.
        """
        ranked = self._rank_by_metric(metric_name)
        if not ranked:
            return []

        k = top_k or max(1, int(len(ranked) * top_ratio))
        top_layers = [name for name, _ in ranked[:k]]

        # Extract module patterns (e.g., "q_proj", "v_proj" from full paths)
        patterns = set()
        for layer_name in top_layers:
            parts = layer_name.split(".")
            # Look for common LoRA target submodule names
            for part in parts:
                if any(
                    kw in part
                    for kw in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        "query",
                        "key",
                        "value",
                        "dense",
                        "fc1",
                        "fc2",
                        "mlp",
                        "attention",
                    )
                ):
                    patterns.add(part)

        # If no specific submodules found, return the layer names
        if not patterns:
            return top_layers

        return sorted(patterns)

    def recommend_lora_rank(
        self,
        metric_name: str = "gradient_norm",
        base_rank: int = 16,
        min_rank: int = 4,
        max_rank: int = 64,
    ) -> int:
        """Recommend a global LoRA rank based on model analysis."""
        ranked = self._rank_by_metric(metric_name)
        if not ranked:
            return base_rank

        scores = [s for _, s in ranked]
        # Higher variance in importance -> higher rank needed
        if len(scores) > 1:
            import numpy as np

            cv = np.std(scores) / (np.mean(scores) + 1e-10)
            # Scale rank with importance variance
            rank = int(base_rank * (1 + cv))
        else:
            rank = base_rank

        return max(min_rank, min(rank, max_rank))

    def generate_config(
        self,
        base_model: str = "",
        metric_name: str = "gradient_norm",
        base_rank: int = 16,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate a complete Axolotl config dict.

        Args:
            base_model: HuggingFace model ID
            metric_name: Metric for importance ranking
            base_rank: Base LoRA rank
            learning_rate: Learning rate
            num_epochs: Training epochs

        Returns:
            Axolotl-compatible config dict
        """
        target_modules = self.recommend_target_modules(metric_name=metric_name)
        rank = self.recommend_lora_rank(metric_name=metric_name, base_rank=base_rank)

        config = {
            "base_model": base_model,
            "model_type": "AutoModelForCausalLM",
            "adapter": "lora",
            "lora_r": rank,
            "lora_alpha": rank * 2,
            "lora_dropout": 0.05,
            "lora_target_modules": target_modules if target_modules else None,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "optimizer": "adamw_torch",
            "lr_scheduler": "cosine",
            "bf16": True,
            "_uni_layer_metadata": {
                "metric_used": metric_name,
                "num_layers_analyzed": len(self.contributions),
                "recommended_rank": rank,
            },
        }

        return config

    def save_yaml(self, path: str, **kwargs) -> str:
        """Generate and save Axolotl YAML config."""
        config = self.generate_config(**kwargs)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return path

    def _rank_by_metric(self, metric_name: str) -> List:
        ranked = []
        for name, metrics in self.contributions.items():
            val = metrics.get(metric_name)
            if val is not None:
                ranked.append((name, val))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


class LLaMAFactoryConfigBridge:
    """
    Generate LLaMA-Factory training config from Uni-Layer analysis.

    Produces JSON configuration with:
    - Adapter type and target modules
    - Layer freezing patterns
    - Training hyperparameters

    Example:
        >>> bridge = LLaMAFactoryConfigBridge(model, contributions)
        >>> config = bridge.generate_config()
        >>> bridge.save_json("llama_factory_config.json")
    """

    def __init__(
        self,
        model: nn.Module,
        contributions: Dict[str, Dict[str, Any]],
    ):
        self.model = model
        self.contributions = contributions

    def recommend_freeze_layers(
        self,
        metric_name: str = "gradient_norm",
        freeze_ratio: float = 0.3,
    ) -> List[str]:
        """
        Recommend layers to freeze (lowest importance).

        Returns layer name patterns for LLaMA-Factory's freeze config.
        """
        ranked = []
        for name, metrics in self.contributions.items():
            val = metrics.get(metric_name)
            if val is not None:
                ranked.append((name, val))
        ranked.sort(key=lambda x: x[1])  # ascending = least important first

        k = max(1, int(len(ranked) * freeze_ratio))
        return [name for name, _ in ranked[:k]]

    def generate_config(
        self,
        model_name: str = "",
        metric_name: str = "gradient_norm",
        base_rank: int = 16,
        finetuning_type: str = "lora",
        learning_rate: float = 5e-5,
        num_train_epochs: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate LLaMA-Factory compatible config.

        Returns:
            Config dict suitable for LLaMA-Factory training
        """
        freeze_layers = self.recommend_freeze_layers(metric_name=metric_name)

        # Determine target modules from non-frozen high-importance layers
        ranked = []
        for name, metrics in self.contributions.items():
            val = metrics.get(metric_name)
            if val is not None:
                ranked.append((name, val))
        ranked.sort(key=lambda x: x[1], reverse=True)

        config = {
            "model_name_or_path": model_name,
            "finetuning_type": finetuning_type,
            "lora_rank": base_rank,
            "lora_alpha": base_rank * 2,
            "lora_dropout": 0.05,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "bf16": True,
            "freeze_layers": freeze_layers,
            "num_frozen_layers": len(freeze_layers),
            "_uni_layer_metadata": {
                "metric_used": metric_name,
                "num_layers_analyzed": len(self.contributions),
                "freeze_ratio": len(freeze_layers) / max(1, len(self.contributions)),
            },
        }

        return config

    def save_json(self, path: str, **kwargs) -> str:
        """Generate and save LLaMA-Factory JSON config."""
        config = self.generate_config(**kwargs)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return path
