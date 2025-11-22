#!/usr/bin/env python
"""
Command-line interface for Uni-Layer.

Usage:
    uni-layer analyze MODEL_PATH --data DATA_PATH --metrics METRICS
    uni-layer benchmark --config CONFIG_FILE
    uni-layer report RESULTS_PATH --output OUTPUT_PATH
"""

import argparse
import sys
from pathlib import Path


def analyze_command(args):
    """Run layer analysis on a model"""
    print(f"Analyzing model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Metrics: {args.metrics}")

    # TODO: Implement actual analysis
    print("\n⚠️  CLI functionality coming soon!")
    print("For now, please use the Python API:")
    print("""
    from uni_layer import LayerAnalyzer
    from uni_layer.metrics import GradientNorm, CKA

    analyzer = LayerAnalyzer(model)
    contributions = analyzer.compute_metrics(
        metrics=[GradientNorm(), CKA()],
        data_loader=data_loader
    )
    """)


def benchmark_command(args):
    """Run benchmark suite"""
    print(f"Running benchmark with config: {args.config}")
    print("\n⚠️  CLI functionality coming soon!")


def report_command(args):
    """Generate report from results"""
    print(f"Generating report from: {args.results}")
    print(f"Output: {args.output}")
    print("\n⚠️  CLI functionality coming soon!")


def list_metrics_command(args):
    """List available metrics"""
    from uni_layer.metrics import (
        GradientNorm, HessianTrace, FisherInformation,
        CKA, EffectiveRank, NTKTrace,
        MutualInformation, ActivationEntropy,
        JacobianRank, DropLayerRobustness,
    )
    from uni_layer.metrics.bayesian import LaplacePosterior
    from uni_layer.metrics.architecture_specific import AttentionFlow

    metrics = [
        ("GradientNorm", "Gradient magnitude measurement"),
        ("HessianTrace", "Hessian trace approximation"),
        ("FisherInformation", "Fisher Information Matrix"),
        ("CKA", "Centered Kernel Alignment"),
        ("EffectiveRank", "Effective rank of representations"),
        ("NTKTrace", "Neural Tangent Kernel trace"),
        ("MutualInformation", "Mutual information with targets"),
        ("ActivationEntropy", "Activation distribution entropy"),
        ("JacobianRank", "Jacobian matrix rank"),
        ("DropLayerRobustness", "Layer importance via ablation"),
        ("LaplacePosterior", "Laplace posterior variance"),
        ("AttentionFlow", "Attention flow analysis (Transformers)"),
    ]

    print("\n📊 Available Metrics:\n")
    print(f"{'Metric':<25} {'Description'}")
    print("="*70)

    for name, desc in metrics:
        print(f"{name:<25} {desc}")

    print("\n💡 Use these metrics with:")
    print("   from uni_layer.metrics import GradientNorm")
    print("   metric = GradientNorm()")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Uni-Layer: Universal Framework for Layer Contribution Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available metrics
  uni-layer list-metrics

  # Analyze a model
  uni-layer analyze model.pt --data data/ --metrics GradientNorm,CKA

  # Run benchmark
  uni-layer benchmark --config benchmark_config.json

  # Generate report
  uni-layer report results.json --output report.html

For more information, visit: https://github.com/GeoffreyWang1117/Uni-Layer
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a model')
    analyze_parser.add_argument('model', help='Path to model file')
    analyze_parser.add_argument('--data', required=True, help='Path to data directory')
    analyze_parser.add_argument('--metrics', default='GradientNorm,CKA',
                               help='Comma-separated list of metrics')
    analyze_parser.add_argument('--output', default='results.json',
                               help='Output file path')
    analyze_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                               help='Device to use')
    analyze_parser.set_defaults(func=analyze_command)

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark suite')
    benchmark_parser.add_argument('--config', required=True,
                                 help='Path to benchmark configuration file')
    benchmark_parser.set_defaults(func=benchmark_command)

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report')
    report_parser.add_argument('results', help='Path to results file')
    report_parser.add_argument('--output', default='report.html',
                              help='Output report path')
    report_parser.add_argument('--format', default='html',
                              choices=['html', 'markdown', 'pdf'],
                              help='Report format')
    report_parser.set_defaults(func=report_command)

    # List metrics command
    list_parser = subparsers.add_parser('list-metrics', help='List available metrics')
    list_parser.set_defaults(func=list_metrics_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
