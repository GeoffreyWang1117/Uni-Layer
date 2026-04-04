#!/usr/bin/env python
"""
MCP (Model Context Protocol) server for Uni-Layer.

Exposes layer analysis tools for integration with Claude Code, Codex, and other
MCP-compatible AI assistants.

Usage:
    python -m uni_layer.mcp_server

Configure in Claude Code settings (~/.claude/settings.json):
    {
      "mcpServers": {
        "uni-layer": {
          "command": "python",
          "args": ["-m", "uni_layer.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "uni-layer",
    instructions="Neural network layer contribution analysis — 26 metrics, 9 categories, 7 architectures",
)

# Cache for loaded models and analysis results to avoid recomputation
_cache: dict[str, Any] = {}


@mcp.tool()
def list_metrics(category: str | None = None) -> str:
    """List all available Uni-Layer metrics.

    Args:
        category: Filter by category (Optimization, Spectral, Information Theory,
                  Representation, Robustness, Bayesian, Efficiency, Security, Architecture).
                  If None, lists all 26 metrics.
    """
    from uni_layer.core.schema import METRIC_OUTPUT_KEYS, METRIC_PRIMARY_KEYS

    categories = {
        "Optimization": ["gradient_norm", "hessian_trace", "fisher_information", "wanda", "ig_sensitivity"],
        "Spectral": ["cka", "effective_rank", "ntk_trace"],
        "Information Theory": ["activation_entropy", "mutual_information"],
        "Representation": ["jacobian_rank", "block_influence"],
        "Robustness": ["droplayer_robustness", "residual_droplayer"],
        "Bayesian": ["laplace_posterior"],
        "Efficiency": ["efficiency", "weight_distribution", "intrinsic_dim", "quantization_sensitivity"],
        "Security": ["adversarial_sensitivity", "activation_anomaly", "membership_inference", "attention_path_trace"],
        "Architecture": ["attention_flow", "moe_router", "diffusion_timestep"],
    }

    if category:
        matched = {k: v for k, v in categories.items() if category.lower() in k.lower()}
        if not matched:
            return f"Unknown category '{category}'. Available: {', '.join(categories.keys())}"
        categories = matched

    result = []
    for cat, metrics in categories.items():
        result.append(f"\n## {cat}")
        for m in metrics:
            pk = METRIC_PRIMARY_KEYS.get(m, m)
            keys = ", ".join(METRIC_OUTPUT_KEYS.get(m, []))
            result.append(f"  - **{m}** (primary: `{pk}`) -> {keys}")

    return "\n".join(result)


@mcp.tool()
def analyze_model(
    model_name: str,
    preset: str = "llm_fast",
    device: str = "cpu",
    num_batches: int = 3,
) -> str:
    """Analyze a HuggingFace model's layers and return per-layer metrics.

    Loads the model, runs the specified preset metrics on dummy data,
    and returns structured per-layer scores.

    Args:
        model_name: HuggingFace model name (e.g. "bert-base-uncased", "meta-llama/Llama-3.2-3B")
        preset: Metric preset — "quick", "llm_fast", "llm_full", or "full"
        device: "cpu" or "cuda"
        num_batches: Number of data batches for metric computation (default 3)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    from uni_layer import LayerAnalyzer

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    texts = ["The quick brown fox jumps over the lazy dog."] * (num_batches * 16)
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    dummy_labels = torch.zeros(encoded["input_ids"].size(0), dtype=torch.long)
    data_loader = DataLoader(
        TensorDataset(encoded["input_ids"], dummy_labels), batch_size=16
    )

    model = model.to(device)
    analyzer = LayerAnalyzer(model, task_type="classification", device=device)
    contributions = analyzer.compute_metrics(
        data_loader=data_loader, preset=preset, num_batches=num_batches,
    )

    # Cache for subsequent tool calls
    _cache["last_model_name"] = model_name
    _cache["last_contributions"] = contributions
    _cache["last_model"] = model
    _cache["last_analyzer"] = analyzer

    # Serialize
    serializable = {}
    for layer_name, metrics in contributions.items():
        serializable[layer_name] = {
            k: round(float(v), 6) if isinstance(v, (int, float)) and v is not None else v
            for k, v in metrics.items()
        }

    return json.dumps(serializable, indent=2, default=str)


@mcp.tool()
def layer_profile(
    model_name: str | None = None,
    target_pruning_ratio: float = 0.3,
    base_lora_rank: int = 8,
) -> str:
    """Generate actionable insights from the last analysis.

    Must call analyze_model first. Returns redundant layers, bottleneck layers,
    pruning suggestions, LoRA suggestions, and a natural language summary.

    Args:
        model_name: Model name for the summary (uses cached name if None)
        target_pruning_ratio: Target ratio for pruning suggestion (0.0-1.0)
        base_lora_rank: Base rank for LoRA suggestion
    """
    contributions = _cache.get("last_contributions")
    if contributions is None:
        return "Error: No analysis results cached. Run analyze_model first."

    from uni_layer import LayerProfile

    name = model_name or _cache.get("last_model_name", "unknown")
    profile = LayerProfile(contributions, model_name=name)

    result = {
        "summary": profile.summary(),
        "redundant_layers": profile.redundant_layers,
        "bottleneck_layers": profile.bottleneck_layers,
        "pruning_suggestion": profile.pruning_suggestion(target_ratio=target_pruning_ratio),
        "lora_suggestion": profile.lora_suggestion(base_rank=base_lora_rank),
        "depth_trends": profile.depth_trends,
        "anomalies": profile.anomalies,
    }

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def suggest_pruning(target_sparsity: float = 0.5, metric: str = "gradient_norm") -> str:
    """Generate Torch-Pruning compatible pruning ratios from the last analysis.

    Args:
        target_sparsity: Overall target sparsity (0.0-1.0)
        metric: Metric to use for importance ranking
    """
    contributions = _cache.get("last_contributions")
    model = _cache.get("last_model")
    if contributions is None or model is None:
        return "Error: No analysis results cached. Run analyze_model first."

    from uni_layer.integrations import TorchPruningBridge

    bridge = TorchPruningBridge(model, contributions)
    ratios = bridge.as_layer_pruning_ratios(metric_name=metric, target_sparsity=target_sparsity)
    protected = bridge.get_protected_layers(top_k=3)

    return json.dumps(
        {"pruning_ratios": ratios, "protected_layers": protected},
        indent=2, default=str,
    )


@mcp.tool()
def suggest_lora(metric: str = "gradient_norm") -> str:
    """Generate PEFT/LoRA configuration from the last analysis.

    Returns recommended target modules and adaptive rank allocation.

    Args:
        metric: Metric to use for layer importance ranking
    """
    contributions = _cache.get("last_contributions")
    model = _cache.get("last_model")
    if contributions is None or model is None:
        return "Error: No analysis results cached. Run analyze_model first."

    from uni_layer.integrations import HuggingFacePEFTBridge

    bridge = HuggingFacePEFTBridge(model, contributions)
    config_params = bridge.recommend_lora_config_params(metric_name=metric)

    return json.dumps(config_params, indent=2, default=str)


@mcp.tool()
def suggest_quantization(target: str = "int8", protect_ratio: float = 0.2) -> str:
    """Generate a mixed-precision quantization plan from the last analysis.

    Args:
        target: Quantization target — "int8" or "fp16"
        protect_ratio: Ratio of most sensitive layers to keep in full precision
    """
    contributions = _cache.get("last_contributions")
    model = _cache.get("last_model")
    if contributions is None or model is None:
        return "Error: No analysis results cached. Run analyze_model first."

    from uni_layer.integrations import ExportHintsBridge

    bridge = ExportHintsBridge(model, contributions)
    plan = bridge.quantization_plan(target=target, protect_ratio=protect_ratio)

    return json.dumps(plan, indent=2, default=str)


@mcp.tool()
def security_audit() -> str:
    """Run a security-focused analysis on the last analyzed model.

    Reports adversarial sensitivity, injection vulnerability,
    membership inference risk, and activation anomalies per layer.
    Must call analyze_model with security metrics first (preset "full"
    or explicit security metrics).
    """
    contributions = _cache.get("last_contributions")
    if contributions is None:
        return "Error: No analysis results cached. Run analyze_model first."

    security_keys = [
        "adv_sensitivity", "adv_amplification", "adv_directional_change",
        "activation_skewness", "activation_kurtosis", "neuron_outlier_ratio",
        "mi_risk_score", "gradient_memorization",
        "injection_vulnerability", "attention_manipulability",
    ]

    report = {}
    has_security = False
    for layer_name, metrics in contributions.items():
        layer_security = {}
        for key in security_keys:
            if key in metrics and metrics[key] is not None:
                layer_security[key] = round(float(metrics[key]), 6)
                has_security = True
        if layer_security:
            report[layer_name] = layer_security

    if not has_security:
        return (
            "No security metrics found in the analysis. "
            "Re-run analyze_model with preset='full' or metrics="
            "'AdversarialSensitivity,ActivationAnomalyScore,MembershipInferenceRisk,AttentionPathTrace'"
        )

    # Find most vulnerable layers
    vuln_layers = []
    for layer_name, metrics in report.items():
        if "injection_vulnerability" in metrics:
            vuln_layers.append((layer_name, metrics["injection_vulnerability"]))
    vuln_layers.sort(key=lambda x: x[1], reverse=True)

    result = {
        "per_layer_security": report,
        "most_vulnerable_layers": vuln_layers[:5] if vuln_layers else "N/A (no AttentionPathTrace data)",
        "total_layers_analyzed": len(report),
    }

    return json.dumps(result, indent=2, default=str)


def main():
    """Run the MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
