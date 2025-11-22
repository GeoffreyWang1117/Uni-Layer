"""
Unit tests for LayerAnalyzer.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.metrics import GradientNorm, CKA


@pytest.fixture
def simple_model():
    """Create a simple model"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )


@pytest.fixture
def sample_data():
    """Create sample dataset"""
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


class TestLayerAnalyzer:
    """Test LayerAnalyzer class"""

    def test_initialization(self, simple_model):
        analyzer = LayerAnalyzer(
            model=simple_model,
            task_type="classification",
            device="cpu"
        )

        assert analyzer.model == simple_model
        assert analyzer.task_type == "classification"
        assert analyzer.device == "cpu"
        assert len(analyzer.layers) > 0

    def test_compute_metrics(self, simple_model, sample_data):
        analyzer = LayerAnalyzer(simple_model, device="cpu")

        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2)],
            data_loader=sample_data,
            verbose=False
        )

        assert len(contributions) > 0
        assert all("gradient_norm" in metrics for metrics in contributions.values())

    def test_rank_layers(self, simple_model, sample_data):
        analyzer = LayerAnalyzer(simple_model, device="cpu")

        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2)],
            data_loader=sample_data,
            verbose=False
        )

        rankings = analyzer.rank_layers(contributions, "gradient_norm")

        assert len(rankings) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in rankings)

        # Check descending order
        values = [v for _, v in rankings]
        assert values == sorted(values, reverse=True)

    def test_get_top_k_layers(self, simple_model, sample_data):
        analyzer = LayerAnalyzer(simple_model, device="cpu")

        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2)],
            data_loader=sample_data,
            verbose=False
        )

        top_layers = analyzer.get_top_k_layers(
            contributions,
            metric_name="gradient_norm",
            k=2
        )

        assert len(top_layers) == 2
        assert all(isinstance(layer, str) for layer in top_layers)

    def test_get_pruning_strategy(self, simple_model, sample_data):
        analyzer = LayerAnalyzer(simple_model, device="cpu")

        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2)],
            data_loader=sample_data,
            verbose=False
        )

        strategy = analyzer.get_pruning_strategy(
            contributions,
            metric_name="gradient_norm",
            prune_ratio=0.3
        )

        assert len(strategy) > 0
        assert all(isinstance(ratio, float) for ratio in strategy.values())
        assert all(0 <= ratio <= 1 for ratio in strategy.values())

    def test_get_distillation_layers(self, simple_model, sample_data):
        analyzer = LayerAnalyzer(simple_model, device="cpu")

        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2)],
            data_loader=sample_data,
            verbose=False
        )

        distill_layers = analyzer.get_distillation_layers(
            contributions,
            metric_name="gradient_norm",
            top_k=3
        )

        assert len(distill_layers) <= 3
        assert all(isinstance(layer, str) for layer in distill_layers)

    def test_aggregate_by_depth(self, simple_model, sample_data):
        analyzer = LayerAnalyzer(simple_model, device="cpu")

        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2)],
            data_loader=sample_data,
            verbose=False
        )

        aggregated = analyzer.aggregate_by_depth(
            contributions,
            metric_name="gradient_norm",
            num_bins=3
        )

        assert len(aggregated) == 3
        assert all(isinstance(v, float) for v in aggregated.values())

    def test_get_summary_statistics(self, simple_model, sample_data):
        analyzer = LayerAnalyzer(simple_model, device="cpu")

        contributions = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2)],
            data_loader=sample_data,
            verbose=False
        )

        stats = analyzer.get_summary_statistics(contributions, "gradient_norm")

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert stats["min"] <= stats["mean"] <= stats["max"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
