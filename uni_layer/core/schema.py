"""
Output format specification for Uni-Layer.

This module defines the canonical output schemas for all Uni-Layer APIs.
Every metric and analyzer method returns data conforming to these schemas.

Output Format Summary
=====================

1. LayerAnalyzer.compute_metrics() -> Dict[str, Dict[str, Any]]
   ---------------------------------------------------------------
   {
     "<layer_name>": {
       "layer_idx":  int,         # Position index in model
       "layer_type": str,         # "linear", "convolution", "transformer_block", ...
       "<metric_key>": float|None # One entry per metric computed
     },
     ...
   }

2. LayerAnalyzer.rank_layers() -> List[Tuple[str, float]]
   -------------------------------------------------------
   [("<layer_name>", <score>), ...]   # Sorted descending by default

3. Individual Metric.compute() -> Dict[str, float]
   ------------------------------------------------
   Each metric returns a flat dict with string keys and float values.
   The primary key shares the metric's name. Additional keys are metric-specific.

Per-Metric Output Keys
======================

Optimization:
  GradientNorm      -> gradient_norm, gradient_norm_std, gradient_norm_max, gradient_norm_min
  HessianTrace      -> hessian_trace, hessian_trace_std
  FisherInformation  -> fisher_information, fisher_mean
  WandaImportance    -> wanda_score, weight_norm, activation_norm, wanda_sparsity
  IGSensitivity      -> ig_sensitivity, ig_variance, ig_relative

Spectral:
  CKA               -> cka_score
  EffectiveRank     -> effective_rank, stable_rank, rank_ratio
  NTKTrace          -> ntk_trace, ntk_trace_per_param

Information Theory:
  ActivationEntropy  -> activation_entropy, activation_mean, activation_std, activation_sparsity
  MutualInformation  -> mutual_information, mi_max, mi_std

Representation:
  JacobianRank      -> jacobian_rank, jacobian_rank_ratio, jacobian_condition, jacobian_max_sv
  BlockInfluence    -> block_influence, block_similarity

Robustness:
  DropLayerRobustness -> droplayer_loss_increase, droplayer_loss_ratio
                        (+ droplayer_acc_decrease, droplayer_acc_ratio if metric="accuracy")
  ResidualDropLayer   -> residual_droplayer_loss_increase, residual_droplayer_loss_ratio,
                        residual_ratio, transform_norm_ratio
                        (+ residual_droplayer_acc_decrease, residual_droplayer_acc_ratio if metric="accuracy")

Bayesian:
  LaplacePosterior  -> laplace_posterior, laplace_posterior_std

Architecture-Specific:
  AttentionFlow     -> attention_entropy, attention_max_weight, head_diversity, attention_distance
  MoERouterAnalysis -> routing_entropy, expert_utilization, load_balance_score,
                       top_expert_ratio, expert_overlap
  DiffusionTimestepAnalysis -> timestep_sensitivity, mean_activation_norm,
                              early_importance, late_importance, timestep_variance
"""

from typing import Any, Dict, List, Optional, Tuple, Union

# Type aliases for documentation and static analysis
MetricResult = Dict[str, Optional[float]]
"""Single metric output: {"metric_key": value, ...}"""

LayerContributions = Dict[str, Dict[str, Any]]
"""Full analysis output: {"layer_name": {"layer_idx": int, "layer_type": str, "metric": float, ...}}"""

LayerRanking = List[Tuple[str, float]]
"""Ranked layers: [("layer_name", score), ...]"""


# Primary keys per metric — the key used for sorting/ranking
METRIC_PRIMARY_KEYS = {
    "gradient_norm": "gradient_norm",
    "hessian_trace": "hessian_trace",
    "fisher_information": "fisher_information",
    "wanda": "wanda_score",
    "ig_sensitivity": "ig_sensitivity",
    "cka": "cka_score",
    "effective_rank": "effective_rank",
    "ntk_trace": "ntk_trace",
    "activation_entropy": "activation_entropy",
    "mutual_information": "mutual_information",
    "jacobian_rank": "jacobian_rank",
    "block_influence": "block_influence",
    "droplayer_robustness": "droplayer_loss_increase",
    "residual_droplayer": "residual_droplayer_loss_increase",
    "laplace_posterior": "laplace_posterior",
    "attention_flow": "attention_entropy",
    "moe_router": "routing_entropy",
    "diffusion_timestep": "timestep_sensitivity",
}

# All output keys per metric
METRIC_OUTPUT_KEYS = {
    "gradient_norm": [
        "gradient_norm",
        "gradient_norm_std",
        "gradient_norm_max",
        "gradient_norm_min",
    ],
    "hessian_trace": ["hessian_trace", "hessian_trace_std"],
    "fisher_information": ["fisher_information", "fisher_mean"],
    "wanda": ["wanda_score", "weight_norm", "activation_norm", "wanda_sparsity"],
    "ig_sensitivity": ["ig_sensitivity", "ig_variance", "ig_relative"],
    "cka": ["cka_score"],
    "effective_rank": ["effective_rank", "stable_rank", "rank_ratio"],
    "ntk_trace": ["ntk_trace", "ntk_trace_per_param"],
    "activation_entropy": [
        "activation_entropy",
        "activation_mean",
        "activation_std",
        "activation_sparsity",
    ],
    "mutual_information": ["mutual_information", "mi_max", "mi_std"],
    "jacobian_rank": [
        "jacobian_rank",
        "jacobian_rank_ratio",
        "jacobian_condition",
        "jacobian_max_sv",
    ],
    "block_influence": ["block_influence", "block_similarity"],
    "droplayer_robustness": ["droplayer_loss_increase", "droplayer_loss_ratio"],
    "residual_droplayer": [
        "residual_droplayer_loss_increase",
        "residual_droplayer_loss_ratio",
        "residual_ratio",
        "transform_norm_ratio",
    ],
    "laplace_posterior": ["laplace_posterior", "laplace_posterior_std"],
    "attention_flow": [
        "attention_entropy",
        "attention_max_weight",
        "head_diversity",
        "attention_distance",
    ],
    "moe_router": [
        "routing_entropy",
        "expert_utilization",
        "load_balance_score",
        "top_expert_ratio",
        "expert_overlap",
    ],
    "diffusion_timestep": [
        "timestep_sensitivity",
        "mean_activation_norm",
        "early_importance",
        "late_importance",
        "timestep_variance",
    ],
}


def validate_metric_result(metric_name: str, result: Dict[str, Any]) -> bool:
    """
    Validate that a metric result conforms to the expected schema.

    Args:
        metric_name: Name of the metric (e.g., "gradient_norm")
        result: The dict returned by metric.compute()

    Returns:
        True if valid

    Raises:
        ValueError: If result is missing required keys or has wrong types
    """
    expected = METRIC_OUTPUT_KEYS.get(metric_name)
    if expected is None:
        raise ValueError(f"Unknown metric: {metric_name}")

    primary = METRIC_PRIMARY_KEYS[metric_name]
    if primary not in result:
        raise ValueError(
            f"Metric '{metric_name}' missing primary key '{primary}'. Got: {list(result.keys())}"
        )

    for key in result:
        val = result[key]
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(
                f"Metric '{metric_name}' key '{key}' has type {type(val).__name__}, expected float|None"
            )

    return True


def validate_contributions(contributions: Dict[str, Dict[str, Any]]) -> bool:
    """
    Validate full compute_metrics() output.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()

    Returns:
        True if valid

    Raises:
        ValueError: If structure is malformed
    """
    if not isinstance(contributions, dict):
        raise ValueError(f"Expected dict, got {type(contributions).__name__}")

    for layer_name, metrics in contributions.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"Layer '{layer_name}': expected dict, got {type(metrics).__name__}")
        if "layer_idx" not in metrics:
            raise ValueError(f"Layer '{layer_name}': missing 'layer_idx'")
        if "layer_type" not in metrics:
            raise ValueError(f"Layer '{layer_name}': missing 'layer_type'")

    return True
