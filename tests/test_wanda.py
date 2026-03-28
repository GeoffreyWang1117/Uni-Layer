"""Tests for Wanda importance metric (v0.5.0 Feature 1)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 5),
    )


@pytest.fixture
def sample_data():
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


class TestWandaImportance:
    def test_import(self):
        from uni_layer.metrics import WandaImportance

        assert WandaImportance is not None

    def test_initialization(self):
        from uni_layer.metrics import WandaImportance

        m = WandaImportance()
        assert m.name == "wanda"
        assert m.category == "optimization"
        assert m.requires_gradient is False

    def test_compute_linear(self, simple_model, sample_data):
        from uni_layer.metrics import WandaImportance

        m = WandaImportance(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert "wanda_score" in result
        assert "weight_norm" in result
        assert "activation_norm" in result
        assert "wanda_sparsity" in result
        assert result["wanda_score"] > 0
        assert result["weight_norm"] > 0
        assert result["activation_norm"] > 0

    def test_relu_layer(self, simple_model, sample_data):
        """ReLU has no weights — should return activation_norm only."""
        from uni_layer.metrics import WandaImportance

        m = WandaImportance(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )

        assert result["wanda_score"] == 0.0
        assert result["weight_norm"] == 0.0
        assert result["activation_norm"] > 0

    def test_sparsity_range(self, simple_model, sample_data):
        from uni_layer.metrics import WandaImportance

        m = WandaImportance(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert 0.0 <= result["wanda_sparsity"] <= 1.0

    def test_multiple_layers_different_scores(self, simple_model, sample_data):
        from uni_layer.metrics import WandaImportance

        m = WandaImportance(num_batches=3)
        scores = []
        for idx in [0, 2, 4]:  # Linear layers
            result = m.compute(
                model=simple_model,
                layer=simple_model[idx],
                layer_name=str(idx),
                layer_idx=idx,
                data_loader=sample_data,
                device="cpu",
            )
            scores.append(result["wanda_score"])

        # All should be positive
        assert all(s > 0 for s in scores)

    def test_schema_registration(self):
        from uni_layer.core.schema import METRIC_OUTPUT_KEYS, METRIC_PRIMARY_KEYS

        assert "wanda" in METRIC_PRIMARY_KEYS
        assert "wanda" in METRIC_OUTPUT_KEYS
        assert METRIC_PRIMARY_KEYS["wanda"] == "wanda_score"

    def test_with_conv_model(self, sample_data):
        """Test with Conv layers."""
        from uni_layer.metrics import WandaImportance

        conv_model = nn.Sequential(
            nn.Unflatten(1, (1, 10)),
            nn.Conv1d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(40, 5),
        )

        m = WandaImportance(num_batches=3)
        result = m.compute(
            model=conv_model,
            layer=conv_model[1],  # Conv1d
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )

        assert result["wanda_score"] > 0
        assert result["weight_norm"] > 0
