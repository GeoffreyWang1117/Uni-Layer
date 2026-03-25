"""
Uni-Layer: A Universal Framework for Layer Contribution Analysis

Uni-Layer provides 30+ layer contribution metrics across 7 theoretical categories,
enabling principled decisions for model compression, fine-tuning, and interpretability.

It is designed to complement — not replace — existing tools like Torch-Pruning,
HuggingFace PEFT, and knowledge distillation frameworks by providing the analytical
foundation ("which layers matter and why?") that drives better optimization decisions.
"""

__version__ = "0.2.1"
__author__ = "Uni-Layer Team"

# Core framework
from uni_layer.core.analyzer import LayerAnalyzer
from uni_layer.core.base_metric import LayerMetric

# Metrics
from uni_layer.metrics import *

# Benchmark
from uni_layer.benchmark.runner import BenchmarkRunner

# Utilities
from uni_layer.utils import ReportGenerator

__all__ = [
    # Core
    "LayerAnalyzer",
    "LayerMetric",
    # Benchmark
    "BenchmarkRunner",
    # Utilities
    "ReportGenerator",
]
