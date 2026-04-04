#!/usr/bin/env python
"""
Command-line interface for Uni-Layer.

Usage:
    uni-layer info                 Show version and environment
    uni-layer list-metrics         List all available metrics
    uni-layer analyze MODEL_NAME   Analyze a HuggingFace model
"""

import argparse
import json
import sys


def cmd_info(args):
    """Show version, environment, and dependency status."""
    import uni_layer

    print(f"uni-layer {uni_layer.__version__}")
    print()

    # Python
    print(f"Python       {sys.version.split()[0]}")

    # PyTorch
    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "N/A"
        print(f"PyTorch      {torch.__version__}")
        print(f"CUDA         {'available (' + device + ')' if cuda else 'not available'}")
    except ImportError:
        print("PyTorch      NOT INSTALLED")

    # Metric count
    from uni_layer.core.schema import METRIC_PRIMARY_KEYS

    print(f"Metrics      {len(METRIC_PRIMARY_KEYS)}")

    # Optional deps
    print()
    print("Optional dependencies:")
    for pkg, label in [
        ("transformers", "HuggingFace Transformers"),
        ("peft", "HuggingFace PEFT"),
        ("torch_pruning", "Torch-Pruning"),
        ("timm", "timm (vision models)"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  {label:<30} {ver}")
        except ImportError:
            print(f"  {label:<30} not installed")


def cmd_list_metrics(args):
    """List all available metrics with category and output keys."""
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

    metric_classes = {
        "gradient_norm": "GradientNorm",
        "hessian_trace": "HessianTrace",
        "fisher_information": "FisherInformation",
        "wanda": "WandaImportance",
        "ig_sensitivity": "IGSensitivity",
        "cka": "CKA",
        "effective_rank": "EffectiveRank",
        "ntk_trace": "NTKTrace",
        "activation_entropy": "ActivationEntropy",
        "mutual_information": "MutualInformation",
        "jacobian_rank": "JacobianRank",
        "block_influence": "BlockInfluence",
        "droplayer_robustness": "DropLayerRobustness",
        "residual_droplayer": "ResidualDropLayer",
        "laplace_posterior": "LaplacePosterior",
        "efficiency": "EfficiencyProfiler",
        "weight_distribution": "WeightDistribution",
        "intrinsic_dim": "IntrinsicDimensionality",
        "quantization_sensitivity": "QuantizationSensitivity",
        "adversarial_sensitivity": "AdversarialSensitivity",
        "activation_anomaly": "ActivationAnomalyScore",
        "membership_inference": "MembershipInferenceRisk",
        "attention_path_trace": "AttentionPathTrace",
        "attention_flow": "AttentionFlow",
        "moe_router": "MoERouterAnalysis",
        "diffusion_timestep": "DiffusionTimestepAnalysis",
    }

    fmt = args.format if hasattr(args, "format") else "table"

    if fmt == "json":
        out = {}
        for cat, metrics in categories.items():
            for m in metrics:
                out[m] = {
                    "class": metric_classes[m],
                    "category": cat,
                    "primary_key": METRIC_PRIMARY_KEYS[m],
                    "output_keys": METRIC_OUTPUT_KEYS[m],
                }
        print(json.dumps(out, indent=2))
        return

    print(f"\n{'Class':<25} {'Category':<20} {'Primary Key':<28} {'All Output Keys'}")
    print("-" * 110)

    for cat, metrics in categories.items():
        for m in metrics:
            cls = metric_classes[m]
            pk = METRIC_PRIMARY_KEYS[m]
            keys = ", ".join(METRIC_OUTPUT_KEYS[m])
            print(f"{cls:<25} {cat:<20} {pk:<28} {keys}")

    print(f"\nTotal: {len(METRIC_PRIMARY_KEYS)} metrics")
    print("\nUsage:")
    print("  from uni_layer.metrics import GradientNorm, CKA, BlockInfluence")


def cmd_analyze(args):
    """Analyze a HuggingFace model or local checkpoint."""
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    model_name = args.model
    device = args.device
    num_batches = args.num_batches

    # Parse metrics
    from uni_layer.metrics import (
        CKA,
        ActivationAnomalyScore,
        ActivationEntropy,
        AdversarialSensitivity,
        AttentionFlow,
        AttentionPathTrace,
        BlockInfluence,
        DiffusionTimestepAnalysis,
        DropLayerRobustness,
        EffectiveRank,
        EfficiencyProfiler,
        FisherInformation,
        GradientNorm,
        HessianTrace,
        IGSensitivity,
        IntrinsicDimensionality,
        JacobianRank,
        LaplacePosterior,
        MembershipInferenceRisk,
        MoERouterAnalysis,
        MutualInformation,
        NTKTrace,
        QuantizationSensitivity,
        ResidualDropLayer,
        WandaImportance,
        WeightDistribution,
    )

    METRIC_MAP = {
        "GradientNorm": GradientNorm,
        "HessianTrace": HessianTrace,
        "FisherInformation": FisherInformation,
        "WandaImportance": WandaImportance,
        "IGSensitivity": IGSensitivity,
        "CKA": CKA,
        "EffectiveRank": EffectiveRank,
        "NTKTrace": NTKTrace,
        "MutualInformation": MutualInformation,
        "ActivationEntropy": ActivationEntropy,
        "JacobianRank": JacobianRank,
        "BlockInfluence": BlockInfluence,
        "DropLayerRobustness": DropLayerRobustness,
        "ResidualDropLayer": ResidualDropLayer,
        "LaplacePosterior": LaplacePosterior,
        "EfficiencyProfiler": EfficiencyProfiler,
        "WeightDistribution": WeightDistribution,
        "IntrinsicDimensionality": IntrinsicDimensionality,
        "QuantizationSensitivity": QuantizationSensitivity,
        "AdversarialSensitivity": AdversarialSensitivity,
        "ActivationAnomalyScore": ActivationAnomalyScore,
        "MembershipInferenceRisk": MembershipInferenceRisk,
        "AttentionPathTrace": AttentionPathTrace,
        "AttentionFlow": AttentionFlow,
        "MoERouterAnalysis": MoERouterAnalysis,
        "DiffusionTimestepAnalysis": DiffusionTimestepAnalysis,
    }

    use_preset = args.preset is not None

    if not use_preset:
        requested = [m.strip() for m in args.metrics.split(",")]
        metric_objs = []
        for name in requested:
            if name not in METRIC_MAP:
                print(
                    f"Error: Unknown metric '{name}'. Available: {', '.join(METRIC_MAP)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            metric_objs.append(METRIC_MAP[name](num_batches=num_batches))

    # Load model
    print(f"Loading model: {model_name}")
    try:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Generate dummy data
        texts = ["The quick brown fox jumps over the lazy dog."] * (num_batches * 16)
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        input_ids = encoded["input_ids"]

        from torch.utils.data import DataLoader, TensorDataset

        dummy_labels = torch.zeros(input_ids.size(0), dtype=torch.long)
        data_loader = DataLoader(TensorDataset(input_ids, dummy_labels), batch_size=16)
        hf_mode = True

    except ImportError:
        print("Error: HuggingFace transformers required for model loading.", file=sys.stderr)
        print("Install with: pip install uni-layer[integrations]", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Analyze
    model = model.to(device)
    from uni_layer import LayerAnalyzer

    analyzer = LayerAnalyzer(model, task_type="classification", device=device)

    if use_preset:
        print(
            f"Running preset '{args.preset}' on {len(analyzer.layers)} layers (batches={num_batches})..."
        )
        contributions = analyzer.compute_metrics(
            data_loader=data_loader,
            preset=args.preset,
            verbose=True,
            num_batches=num_batches,
        )
    else:
        print(
            f"Running {len(metric_objs)} metrics on {len(analyzer.layers)} layers (batches={num_batches})..."
        )
        contributions = analyzer.compute_metrics(
            metrics=metric_objs,
            data_loader=data_loader,
            verbose=True,
            num_batches=num_batches,
        )

    # LayerProfile insights
    if args.profile:
        from uni_layer import LayerProfile

        profile = LayerProfile(contributions, model_name=model_name)
        print(f"\n{'=' * 70}")
        print("LAYER PROFILE SUMMARY")
        print(f"{'=' * 70}")
        print(profile.summary())
        print(f"\nRedundant layers: {profile.redundant_layers}")
        print(f"Bottleneck layers: {profile.bottleneck_layers}")
        pruning = profile.pruning_suggestion(target_ratio=0.3)
        print(f"\nPruning suggestion (30%): {json.dumps(pruning, indent=2, default=str)}")
        lora = profile.lora_suggestion(base_rank=8)
        print(f"\nLoRA suggestion (rank=8): {json.dumps(lora, indent=2, default=str)}")

    # Output
    if args.output:
        serializable = {}
        for layer_name, metrics in contributions.items():
            serializable[layer_name] = {
                k: float(v) if isinstance(v, (int, float)) and v is not None else v
                for k, v in metrics.items()
            }
        if args.profile:
            serializable["_profile"] = profile.to_dict()
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    elif not args.profile:
        # Print table to stdout (skip if --profile already printed insights)
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        # Collect all metric keys present in the results
        all_metric_keys = set()
        for layer_metrics in contributions.values():
            all_metric_keys.update(
                k for k in layer_metrics if k not in ("layer_idx", "layer_type")
            )

        # Use primary keys from schema for display
        pk_names = sorted(
            k for k in all_metric_keys if k in METRIC_PRIMARY_KEYS.values()
        )
        if not pk_names:
            pk_names = sorted(all_metric_keys)

        header = f"{'Layer':<35} {'Type':<20}"
        for pk in pk_names:
            header += f" {pk:>15}"
        print(f"\n{header}")
        print("-" * len(header))

        for layer_name, metrics in contributions.items():
            row = f"{layer_name:<35} {metrics.get('layer_type', '?'):<20}"
            has_data = False
            for pk in pk_names:
                val = metrics.get(pk)
                if val is not None:
                    row += f" {val:>15.6f}"
                    has_data = True
                else:
                    row += f" {'N/A':>15}"
            if has_data:
                print(row)

        # Ranking by first primary key
        if pk_names:
            rank_key = pk_names[0]
            ranked = analyzer.rank_layers(contributions, rank_key)
            if ranked:
                print(f"\nRanking by {rank_key} (descending):")
                for i, (name, score) in enumerate(ranked[:10]):
                    print(f"  {i+1:>3}. {name:<35} {score:.6f}")
                if len(ranked) > 10:
                    print(f"  ... and {len(ranked) - 10} more layers")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="uni-layer",
        description="Uni-Layer: Layer Contribution Analysis for Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uni-layer info                              Show version and environment
  uni-layer list-metrics                      List all metrics
  uni-layer list-metrics --format json        Output as JSON
  uni-layer analyze bert-base-uncased         Analyze a HuggingFace model
  uni-layer analyze bert-base-uncased -m GradientNorm,BlockInfluence -o results.json

Documentation: https://github.com/GeoffreyWang1117/Uni-Layer""",
    )

    parser.add_argument("--version", action="store_true", help="Show version")

    sub = parser.add_subparsers(dest="command")

    # info
    sub.add_parser("info", help="Show version, environment, and dependency status")

    # list-metrics
    lm = sub.add_parser("list-metrics", help="List all available metrics")
    lm.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # analyze
    az = sub.add_parser("analyze", help="Analyze a HuggingFace model")
    az.add_argument("model", help="HuggingFace model name or path (e.g. bert-base-uncased)")
    az.add_argument(
        "-m",
        "--metrics",
        default="GradientNorm,BlockInfluence,EffectiveRank",
        help="Comma-separated metrics (default: GradientNorm,BlockInfluence,EffectiveRank)",
    )
    az.add_argument(
        "-p",
        "--preset",
        choices=["quick", "llm_fast", "llm_full", "full"],
        default=None,
        help="Use a preset instead of -m (overrides --metrics)",
    )
    az.add_argument(
        "--profile",
        action="store_true",
        help="Generate LayerProfile summary with actionable insights",
    )
    az.add_argument(
        "-o", "--output", default=None, help="Save results to JSON file (default: print table)"
    )
    az.add_argument(
        "-d", "--device", default="cpu", choices=["cpu", "cuda"], help="Device (default: cpu)"
    )
    az.add_argument(
        "-n", "--num-batches", type=int, default=3, help="Number of data batches (default: 3)"
    )

    args = parser.parse_args()

    if args.version:
        import uni_layer

        print(f"uni-layer {uni_layer.__version__}")
        return

    if args.command == "info":
        cmd_info(args)
    elif args.command == "list-metrics":
        cmd_list_metrics(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()
        sys.exit(0 if not args.command else 1)


if __name__ == "__main__":
    main()
