"""Tests for efficiency metrics (v0.6.1 Features 1-4)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 5)
    )


@pytest.fixture
def sample_data():
    X = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


# ──────────────────── EfficiencyProfiler ────────────────────


class TestEfficiencyProfiler:
    def test_import(self):
        from uni_layer.metrics import EfficiencyProfiler

        assert EfficiencyProfiler is not None

    def test_init(self):
        from uni_layer.metrics import EfficiencyProfiler

        m = EfficiencyProfiler()
        assert m.name == "efficiency"
        assert m.category == "efficiency"

    def test_compute_linear(self, simple_model, sample_data):
        from uni_layer.metrics import EfficiencyProfiler

        EfficiencyProfiler.clear_cache()
        m = EfficiencyProfiler()
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["flops"] > 0
        assert result["param_count"] == 10 * 20 + 20  # weight + bias
        assert result["param_memory_mb"] > 0
        assert result["activation_memory_mb"] > 0
        assert 0.0 <= result["compute_ratio"] <= 1.0

    def test_relu_layer(self, simple_model, sample_data):
        from uni_layer.metrics import EfficiencyProfiler

        EfficiencyProfiler.clear_cache()
        m = EfficiencyProfiler()
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["param_count"] == 0
        assert result["flops"] == 0

    def test_compute_ratio_sums(self, simple_model, sample_data):
        """Compute ratios across all layers should approximately sum to ~1."""
        from uni_layer.metrics import EfficiencyProfiler

        EfficiencyProfiler.clear_cache()
        m = EfficiencyProfiler()
        ratios = []
        for idx in range(5):
            result = m.compute(
                model=simple_model,
                layer=simple_model[idx],
                layer_name=str(idx),
                layer_idx=idx,
                data_loader=sample_data,
                device="cpu",
            )
            ratios.append(result["compute_ratio"])
        total = sum(ratios)
        assert 0.5 <= total <= 1.5  # approximately 1.0

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "efficiency" in METRIC_PRIMARY_KEYS


# ──────────────────── WeightDistribution ────────────────────


class TestWeightDistribution:
    def test_import(self):
        from uni_layer.metrics import WeightDistribution

        assert WeightDistribution is not None

    def test_init(self):
        from uni_layer.metrics import WeightDistribution

        m = WeightDistribution()
        assert m.name == "weight_distribution"
        assert m.requires_data is False

    def test_compute(self, simple_model):
        from uni_layer.metrics import WeightDistribution

        m = WeightDistribution()
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            device="cpu",
        )
        assert "weight_sparsity" in result
        assert "weight_l1_norm" in result
        assert "weight_l2_norm" in result
        assert "weight_rank_ratio" in result
        assert "weight_outlier_ratio" in result
        assert "weight_kurtosis" in result

    def test_sparsity_range(self, simple_model):
        from uni_layer.metrics import WeightDistribution

        m = WeightDistribution()
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            device="cpu",
        )
        assert 0.0 <= result["weight_sparsity"] <= 1.0
        assert 0.0 <= result["weight_outlier_ratio"] <= 1.0
        assert 0.0 <= result["weight_rank_ratio"] <= 1.0 + 1e-6

    def test_norms_positive(self, simple_model):
        from uni_layer.metrics import WeightDistribution

        m = WeightDistribution()
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            device="cpu",
        )
        assert result["weight_l1_norm"] > 0
        assert result["weight_l2_norm"] > 0

    def test_no_params_layer(self, simple_model):
        from uni_layer.metrics import WeightDistribution

        m = WeightDistribution()
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],
            layer_name="1",
            layer_idx=1,
            device="cpu",
        )
        assert result["weight_sparsity"] == 0.0

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "weight_distribution" in METRIC_PRIMARY_KEYS


# ──────────────────── IntrinsicDimensionality ────────────────────


class TestIntrinsicDimensionality:
    def test_import(self):
        from uni_layer.metrics import IntrinsicDimensionality

        assert IntrinsicDimensionality is not None

    def test_init(self):
        from uni_layer.metrics import IntrinsicDimensionality

        m = IntrinsicDimensionality()
        assert m.name == "intrinsic_dim"
        assert m.requires_data is True

    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import IntrinsicDimensionality

        m = IntrinsicDimensionality(num_batches=3, k=10)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "intrinsic_dim" in result
        assert "intrinsic_dim_ratio" in result
        assert "ambient_dim" in result
        assert result["intrinsic_dim"] >= 0
        assert result["ambient_dim"] == 20  # output of Linear(10, 20)

    def test_ratio_range(self, simple_model, sample_data):
        from uni_layer.metrics import IntrinsicDimensionality

        m = IntrinsicDimensionality(num_batches=3, k=10)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert 0.0 <= result["intrinsic_dim_ratio"] <= 2.0  # can exceed 1 slightly

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "intrinsic_dim" in METRIC_PRIMARY_KEYS


# ──────────────────── QuantizationSensitivity ────────────────────


class TestQuantizationSensitivity:
    def test_import(self):
        from uni_layer.metrics import QuantizationSensitivity

        assert QuantizationSensitivity is not None

    def test_init(self):
        from uni_layer.metrics import QuantizationSensitivity

        m = QuantizationSensitivity()
        assert m.name == "quantization_sensitivity"
        assert m.requires_data is True

    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import QuantizationSensitivity

        m = QuantizationSensitivity(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "quant_sensitivity_int8" in result
        assert "quant_sensitivity_fp16" in result
        assert "activation_range" in result
        assert "weight_dynamic_range" in result

    def test_int8_more_sensitive_than_fp16(self, simple_model, sample_data):
        """INT8 should generally cause more deviation than FP16."""
        from uni_layer.metrics import QuantizationSensitivity

        m = QuantizationSensitivity(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["quant_sensitivity_int8"] >= result["quant_sensitivity_fp16"]

    def test_sensitivities_nonneg(self, simple_model, sample_data):
        from uni_layer.metrics import QuantizationSensitivity

        m = QuantizationSensitivity(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["quant_sensitivity_int8"] >= 0
        assert result["quant_sensitivity_fp16"] >= 0

    def test_no_params_layer(self, simple_model, sample_data):
        from uni_layer.metrics import QuantizationSensitivity

        m = QuantizationSensitivity(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["quant_sensitivity_int8"] == 0.0

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "quantization_sensitivity" in METRIC_PRIMARY_KEYS
