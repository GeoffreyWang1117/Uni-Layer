"""Tests for IG Sensitivity metric (v0.5.0 Feature 2)."""

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


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


class TestIGSensitivity:
    def test_import(self):
        from uni_layer.metrics import IGSensitivity

        assert IGSensitivity is not None

    def test_initialization(self):
        from uni_layer.metrics import IGSensitivity

        m = IGSensitivity()
        assert m.name == "ig_sensitivity"
        assert m.category == "optimization"
        assert m.requires_gradient is True

    def test_compute_linear(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import IGSensitivity

        m = IGSensitivity(num_steps=5, num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert "ig_sensitivity" in result
        assert "ig_variance" in result
        assert "ig_relative" in result
        assert result["ig_sensitivity"] >= 0
        assert result["ig_variance"] >= 0

    def test_no_params_layer(self, simple_model, sample_data, criterion):
        """ReLU has no parameters — should return zeros."""
        from uni_layer.metrics import IGSensitivity

        m = IGSensitivity(num_steps=3, num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        assert result["ig_sensitivity"] == 0.0
        assert result["ig_variance"] == 0.0

    def test_multiple_layers(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import IGSensitivity

        m = IGSensitivity(num_steps=3, num_batches=2)
        scores = []
        for idx in [0, 2, 4]:
            result = m.compute(
                model=simple_model,
                layer=simple_model[idx],
                layer_name=str(idx),
                layer_idx=idx,
                data_loader=sample_data,
                device="cpu",
                criterion=criterion,
            )
            scores.append(result["ig_sensitivity"])

        assert all(s >= 0 for s in scores)

    def test_ig_relative_scales_with_params(self, simple_model, sample_data, criterion):
        """ig_relative should be ig_sensitivity * num_params."""
        from uni_layer.metrics import IGSensitivity

        m = IGSensitivity(num_steps=3, num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )

        num_params = sum(p.numel() for p in simple_model[0].parameters())
        expected = result["ig_sensitivity"] * num_params
        assert abs(result["ig_relative"] - expected) < 1e-4

    def test_schema_registration(self):
        from uni_layer.core.schema import METRIC_OUTPUT_KEYS, METRIC_PRIMARY_KEYS

        assert "ig_sensitivity" in METRIC_PRIMARY_KEYS
        assert "ig_sensitivity" in METRIC_OUTPUT_KEYS
