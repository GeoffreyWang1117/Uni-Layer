"""
Tests for MoE Router Analysis metric (v0.4.0 Feature 3).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleGate(nn.Module):
    """Simple gating network that produces expert selection logits."""

    def __init__(self, dim, num_experts):
        super().__init__()
        self.linear = nn.Linear(dim, num_experts)

    def forward(self, x):
        return self.linear(x)


class SimpleExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class SimpleMoELayer(nn.Module):
    """Minimal MoE layer for testing."""

    def __init__(self, dim=10, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = SimpleGate(dim, num_experts)
        self.experts = nn.ModuleList([SimpleExpert(dim) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, indices = torch.topk(torch.softmax(gate_logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = weights[mask][:, (indices[mask] == i).nonzero(as_tuple=True)[1][:1]].mean(
                    dim=-1, keepdim=True
                )
                output[mask] += expert_out * weight
        return output


class MoEModel(nn.Module):
    """Full model with MoE layers for testing."""

    def __init__(self, dim=10, num_experts=4, num_layers=2, num_classes=5):
        super().__init__()
        self.moe_layers = nn.ModuleList(
            [SimpleMoELayer(dim, num_experts) for _ in range(num_layers)]
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        for layer in self.moe_layers:
            x = x + layer(x)
        return self.head(x)


@pytest.fixture
def moe_model():
    return MoEModel(dim=10, num_experts=4, num_layers=2, num_classes=5)


@pytest.fixture
def non_moe_model():
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def sample_data():
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


class TestMoERouterAnalysis:
    def test_import(self):
        from uni_layer.metrics import MoERouterAnalysis

        assert MoERouterAnalysis is not None

    def test_initialization(self):
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis()
        assert m.name == "moe_router"
        assert m.category == "architecture_specific"
        assert m.requires_gradient is False

    def test_non_moe_layer_returns_none(self, non_moe_model, sample_data):
        """Non-MoE layers should return None values."""
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis(num_batches=2)
        result = m.compute(
            model=non_moe_model,
            layer=non_moe_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert result["routing_entropy"] is None
        assert result["expert_utilization"] is None
        assert result["load_balance_score"] is None

    def test_compute_moe_layer(self, moe_model, sample_data):
        """MoE layers should return valid routing metrics."""
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis(num_batches=3)
        moe_layer = moe_model.moe_layers[0]

        result = m.compute(
            model=moe_model,
            layer=moe_layer,
            layer_name="moe_layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert result["routing_entropy"] is not None
        assert result["expert_utilization"] is not None
        assert result["load_balance_score"] is not None
        assert result["top_expert_ratio"] is not None
        assert result["expert_overlap"] is not None

    def test_routing_entropy_range(self, moe_model, sample_data):
        """Normalized routing entropy should be in [0, 1]."""
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis(num_batches=3)
        result = m.compute(
            model=moe_model,
            layer=moe_model.moe_layers[0],
            layer_name="moe_layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert 0.0 <= result["routing_entropy"] <= 1.0 + 1e-6

    def test_expert_utilization_range(self, moe_model, sample_data):
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis(num_batches=3)
        result = m.compute(
            model=moe_model,
            layer=moe_model.moe_layers[0],
            layer_name="moe_layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert 0.0 <= result["expert_utilization"] <= 1.0

    def test_top_expert_ratio_range(self, moe_model, sample_data):
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis(num_batches=3)
        result = m.compute(
            model=moe_model,
            layer=moe_model.moe_layers[0],
            layer_name="moe_layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert 0.0 <= result["top_expert_ratio"] <= 1.0

    def test_load_balance_score_nonneg(self, moe_model, sample_data):
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis(num_batches=3)
        result = m.compute(
            model=moe_model,
            layer=moe_model.moe_layers[0],
            layer_name="moe_layers.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )

        assert result["load_balance_score"] >= 0.0

    def test_multiple_moe_layers(self, moe_model, sample_data):
        """Both MoE layers should produce valid results."""
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis(num_batches=3)
        results = []
        for i, layer in enumerate(moe_model.moe_layers):
            result = m.compute(
                model=moe_model,
                layer=layer,
                layer_name=f"moe_layers.{i}",
                layer_idx=i,
                data_loader=sample_data,
                device="cpu",
            )
            results.append(result)
            assert result["routing_entropy"] is not None

    def test_is_moe_layer_detection(self):
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis()

        moe_layer = SimpleMoELayer(dim=10, num_experts=4)
        assert m._is_moe_layer(moe_layer)

        linear = nn.Linear(10, 5)
        assert not m._is_moe_layer(linear)

    def test_get_num_experts(self):
        from uni_layer.metrics import MoERouterAnalysis

        m = MoERouterAnalysis()
        moe_layer = SimpleMoELayer(dim=10, num_experts=4)
        assert m._get_num_experts(moe_layer) == 4

    def test_schema_registration(self):
        from uni_layer.core.schema import METRIC_OUTPUT_KEYS, METRIC_PRIMARY_KEYS

        assert "moe_router" in METRIC_PRIMARY_KEYS
        assert "moe_router" in METRIC_OUTPUT_KEYS
        assert METRIC_PRIMARY_KEYS["moe_router"] == "routing_entropy"
