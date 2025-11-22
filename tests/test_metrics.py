"""
Unit tests for layer contribution metrics.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer.metrics import (
    GradientNorm,
    CKA,
    EffectiveRank,
    ActivationEntropy,
    DropLayerRobustness,
)


@pytest.fixture
def simple_model():
    """Create a simple model for testing"""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    return model


@pytest.fixture
def sample_data():
    """Create sample dataset"""
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


class TestGradientNorm:
    """Test GradientNorm metric"""

    def test_initialization(self):
        metric = GradientNorm()
        assert metric.name == "gradient_norm"
        assert metric.category == "optimization"
        assert metric.requires_gradient == True
        assert metric.requires_data == True

    def test_compute(self, simple_model, sample_data):
        metric = GradientNorm(num_batches=2)
        layer = simple_model[0]  # First linear layer

        result = metric.compute(
            model=simple_model,
            layer=layer,
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=nn.CrossEntropyLoss()
        )

        assert "gradient_norm" in result
        assert isinstance(result["gradient_norm"], float)
        assert result["gradient_norm"] >= 0

    def test_different_norm_types(self, simple_model, sample_data):
        for norm_type in ["l1", "l2", "linf"]:
            metric = GradientNorm(norm_type=norm_type, num_batches=1)
            layer = simple_model[0]

            result = metric.compute(
                model=simple_model,
                layer=layer,
                layer_name="0",
                layer_idx=0,
                data_loader=sample_data,
                device="cpu"
            )

            assert result["gradient_norm"] >= 0


class TestCKA:
    """Test CKA metric"""

    def test_initialization(self):
        metric = CKA()
        assert metric.name == "cka"
        assert metric.category == "spectral"
        assert metric.requires_gradient == False

    def test_compute(self, simple_model, sample_data):
        metric = CKA(num_batches=2)
        layer = simple_model[0]

        result = metric.compute(
            model=simple_model,
            layer=layer,
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu"
        )

        assert "cka_score" in result
        assert 0 <= result["cka_score"] <= 1  # CKA is bounded [0, 1]


class TestEffectiveRank:
    """Test EffectiveRank metric"""

    def test_initialization(self):
        metric = EffectiveRank()
        assert metric.name == "effective_rank"
        assert metric.category == "spectral"

    def test_compute(self, simple_model, sample_data):
        metric = EffectiveRank(num_batches=2)
        layer = simple_model[0]

        result = metric.compute(
            model=simple_model,
            layer=layer,
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu"
        )

        assert "effective_rank" in result
        assert result["effective_rank"] >= 0
        assert "stable_rank" in result
        assert "rank_ratio" in result


class TestActivationEntropy:
    """Test ActivationEntropy metric"""

    def test_compute(self, simple_model, sample_data):
        metric = ActivationEntropy(num_batches=2)
        layer = simple_model[0]

        result = metric.compute(
            model=simple_model,
            layer=layer,
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu"
        )

        assert "activation_entropy" in result
        assert result["activation_entropy"] >= 0
        assert "activation_sparsity" in result
        assert 0 <= result["activation_sparsity"] <= 1


class TestDropLayerRobustness:
    """Test DropLayerRobustness metric"""

    def test_compute(self, simple_model, sample_data):
        metric = DropLayerRobustness(num_batches=2)
        layer = simple_model[0]

        result = metric.compute(
            model=simple_model,
            layer=layer,
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=nn.CrossEntropyLoss()
        )

        assert "droplayer_loss_increase" in result
        assert isinstance(result["droplayer_loss_increase"], float)


def test_metric_base_class():
    """Test LayerMetric base class"""
    from uni_layer.core.base_metric import LayerMetric

    # Should not be able to instantiate abstract class directly
    with pytest.raises(TypeError):
        LayerMetric(name="test", category="test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
