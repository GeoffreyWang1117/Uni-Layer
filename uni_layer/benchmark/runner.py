"""
Benchmark evaluation pipeline for layer contribution analysis.

This module provides tools for systematic benchmarking across models.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from uni_layer import LayerAnalyzer
from uni_layer.core.base_metric import LayerMetric


class BenchmarkRunner:
    """
    Run systematic benchmarks across models and metrics.

    This class helps create reproducible benchmarks by:
    - Running multiple metrics on multiple models
    - Saving results in a structured format
    - Comparing results across runs
    - Generating benchmark reports

    Example:
        >>> runner = BenchmarkRunner(save_dir="benchmarks/")
        >>> runner.add_model("bert", bert_model, bert_loader)
        >>> runner.add_model("gpt2", gpt2_model, gpt2_loader)
        >>> runner.run(metrics=[GradientNorm(), CKA()])
        >>> runner.save_results()
    """

    def __init__(
        self,
        save_dir: str = "benchmarks",
        run_name: Optional[str] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            save_dir: Directory to save benchmark results
            run_name: Name for this benchmark run (auto-generated if None)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run_name = run_name
        self.run_dir = self.save_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.results = {}
        self.metadata = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "metrics": [],
        }

    def add_model(
        self,
        model_name: str,
        model: nn.Module,
        data_loader: DataLoader,
        task_type: str = "classification",
        device: str = "cuda",
    ):
        """
        Add a model to the benchmark.

        Args:
            model_name: Unique name for the model
            model: PyTorch model
            data_loader: DataLoader for evaluation
            task_type: Task type
            device: Device to run on
        """
        self.models[model_name] = {
            "model": model,
            "data_loader": data_loader,
            "task_type": task_type,
            "device": device,
        }

        # Store metadata
        self.metadata["models"][model_name] = {
            "task_type": task_type,
            "device": device,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_layers": len(list(model.modules())),
        }

        print(f"✓ Added model '{model_name}' to benchmark")

    def run(
        self,
        metrics: List[LayerMetric],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run benchmark on all models.

        Args:
            metrics: List of metrics to compute
            verbose: Whether to show progress

        Returns:
            Nested dictionary: {model_name: {layer_name: {metric_name: value}}}
        """
        print(f"\n{'='*60}")
        print(f"Running Benchmark: {self.run_name}")
        print(f"{'='*60}\n")

        # Store metric names
        self.metadata["metrics"] = [m.name for m in metrics]

        # Run on each model
        for model_name, model_config in self.models.items():
            print(f"\n[{model_name}] Starting analysis...")

            # Create analyzer
            analyzer = LayerAnalyzer(
                model=model_config["model"],
                task_type=model_config["task_type"],
                device=model_config["device"],
            )

            # Compute metrics
            contributions = analyzer.compute_metrics(
                metrics=metrics,
                data_loader=model_config["data_loader"],
                verbose=verbose,
            )

            # Store results
            self.results[model_name] = contributions

            print(f"✓ [{model_name}] Analysis complete")
            print(f"  Analyzed {len(contributions)} layers")

        print(f"\n{'='*60}")
        print("Benchmark Complete!")
        print(f"{'='*60}\n")

        return self.results

    def save_results(self, format: str = "json"):
        """
        Save benchmark results.

        Args:
            format: Save format ('json' or 'pickle')
        """
        # Save results
        if format == "json":
            results_file = self.run_dir / "results.json"
            with open(results_file, "w") as f:
                # Convert to JSON-serializable format
                json_results = {}
                for model_name, model_results in self.results.items():
                    json_results[model_name] = {}
                    for layer_name, metrics in model_results.items():
                        json_results[model_name][layer_name] = {
                            k: (float(v) if v is not None else None) for k, v in metrics.items()
                        }

                json.dump(json_results, f, indent=2)

            print(f"✓ Results saved to {results_file}")

        elif format == "pickle":
            results_file = self.run_dir / "results.pkl"
            with open(results_file, "wb") as f:
                pickle.dump(self.results, f)

            print(f"✓ Results saved to {results_file}")

        # Save metadata
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✓ Metadata saved to {metadata_file}")

    def load_results(self, run_name: str, format: str = "json"):
        """
        Load previously saved results.

        Args:
            run_name: Name of the benchmark run
            format: Load format ('json' or 'pickle')
        """
        run_dir = self.save_dir / run_name

        if format == "json":
            results_file = run_dir / "results.json"
            with open(results_file, "r") as f:
                self.results = json.load(f)

        elif format == "pickle":
            results_file = run_dir / "results.pkl"
            with open(results_file, "rb") as f:
                self.results = pickle.load(f)

        # Load metadata
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        print(f"✓ Loaded results from {run_name}")

    def compare_models(
        self,
        metric_name: str,
        models: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compare models on a specific metric.

        Args:
            metric_name: Name of the metric to compare
            models: List of model names (default: all models)

        Returns:
            Dictionary mapping model names to average metric values
        """
        if models is None:
            models = list(self.results.keys())

        comparison = {}

        for model_name in models:
            if model_name not in self.results:
                continue

            values = []
            for layer_metrics in self.results[model_name].values():
                if metric_name in layer_metrics and layer_metrics[metric_name] is not None:
                    values.append(layer_metrics[metric_name])

            if values:
                comparison[model_name] = sum(values) / len(values)

        return comparison

    def generate_report(self) -> str:
        """
        Generate a text report of the benchmark.

        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Benchmark Report: {self.run_name}")
        lines.append("=" * 60)
        lines.append(f"\nTimestamp: {self.metadata['timestamp']}")
        lines.append(f"Models: {len(self.metadata['models'])}")
        lines.append(f"Metrics: {len(self.metadata['metrics'])}")

        lines.append("\n" + "=" * 60)
        lines.append("Model Summary")
        lines.append("=" * 60)

        for model_name, model_meta in self.metadata["models"].items():
            lines.append(f"\n{model_name}:")
            lines.append(f"  Task: {model_meta['task_type']}")
            lines.append(f"  Parameters: {model_meta['num_parameters']:,}")
            lines.append(f"  Layers analyzed: {len(self.results.get(model_name, {}))}")

        lines.append("\n" + "=" * 60)
        lines.append("Metric Comparison")
        lines.append("=" * 60)

        for metric_name in self.metadata["metrics"]:
            lines.append(f"\n{metric_name}:")
            comparison = self.compare_models(metric_name)

            for model_name, avg_value in sorted(
                comparison.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {model_name}: {avg_value:.4f}")

        report = "\n".join(lines)

        # Save report
        report_file = self.run_dir / "report.txt"
        with open(report_file, "w") as f:
            f.write(report)

        print(f"✓ Report saved to {report_file}")

        return report


# Convenience function
def run_quick_benchmark(
    models: Dict[str, tuple],
    metrics: List[LayerMetric],
    save_dir: str = "benchmarks",
) -> BenchmarkRunner:
    """
    Quick benchmark multiple models.

    Args:
        models: Dictionary {model_name: (model, data_loader, task_type)}
        metrics: List of metrics to compute
        save_dir: Directory to save results

    Returns:
        BenchmarkRunner with results

    Example:
        >>> results = run_quick_benchmark(
        ...     models={
        ...         "bert": (bert_model, bert_loader, "classification"),
        ...         "gpt2": (gpt2_model, gpt2_loader, "generation"),
        ...     },
        ...     metrics=[GradientNorm(), CKA()]
        ... )
    """
    runner = BenchmarkRunner(save_dir=save_dir)

    for model_name, (model, data_loader, task_type) in models.items():
        runner.add_model(model_name, model, data_loader, task_type)

    runner.run(metrics=metrics)
    runner.save_results()
    runner.generate_report()

    return runner
