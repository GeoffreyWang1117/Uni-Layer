"""Tests for security/red-team metrics (v0.6.0 Features 1-4)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleTransformerBlock(nn.Module):
    """Minimal block with attention for testing."""

    def __init__(self, dim=32, nhead=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + attn_out)
        return x + self.mlp(x)


class TransformerModel(nn.Module):
    def __init__(self, dim=32, nhead=4, num_layers=3, num_classes=5):
        super().__init__()
        self.embed = nn.Linear(10, dim)
        self.blocks = nn.ModuleList([SimpleTransformerBlock(dim, nhead) for _ in range(num_layers)])
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1).expand(-1, 8, -1)  # fake seq
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 5)
    )


@pytest.fixture
def transformer_model():
    return TransformerModel(dim=32, nhead=4, num_layers=3, num_classes=5)


@pytest.fixture
def sample_data():
    X = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


# ──────────────────── AdversarialSensitivity ────────────────────


class TestAdversarialSensitivity:
    def test_import(self):
        from uni_layer.metrics import AdversarialSensitivity

        assert AdversarialSensitivity is not None

    def test_init(self):
        from uni_layer.metrics import AdversarialSensitivity

        m = AdversarialSensitivity()
        assert m.name == "adversarial_sensitivity"
        assert m.category == "security"
        assert m.requires_gradient is True

    def test_compute(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import AdversarialSensitivity

        m = AdversarialSensitivity(epsilon=0.01, num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert "adv_sensitivity" in result
        assert "adv_amplification" in result
        assert "adv_directional_change" in result
        assert result["adv_sensitivity"] >= 0

    def test_sensitivity_nonneg(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import AdversarialSensitivity

        m = AdversarialSensitivity(epsilon=0.01, num_batches=2)
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
            assert result["adv_sensitivity"] >= 0
            assert result["adv_directional_change"] >= 0

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "adversarial_sensitivity" in METRIC_PRIMARY_KEYS


# ──────────────────── ActivationAnomalyScore ────────────────────


class TestActivationAnomalyScore:
    def test_import(self):
        from uni_layer.metrics import ActivationAnomalyScore

        assert ActivationAnomalyScore is not None

    def test_init(self):
        from uni_layer.metrics import ActivationAnomalyScore

        m = ActivationAnomalyScore()
        assert m.name == "activation_anomaly"
        assert m.category == "security"
        assert m.requires_gradient is False

    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import ActivationAnomalyScore

        m = ActivationAnomalyScore(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "activation_skewness" in result
        assert "activation_kurtosis" in result
        assert "neuron_outlier_ratio" in result
        assert "activation_bimodality" in result

    def test_outlier_ratio_range(self, simple_model, sample_data):
        from uni_layer.metrics import ActivationAnomalyScore

        m = ActivationAnomalyScore(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert 0.0 <= result["neuron_outlier_ratio"] <= 1.0

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "activation_anomaly" in METRIC_PRIMARY_KEYS


# ──────────────────── MembershipInferenceRisk ────────────────────


class TestMembershipInferenceRisk:
    def test_import(self):
        from uni_layer.metrics import MembershipInferenceRisk

        assert MembershipInferenceRisk is not None

    def test_init(self):
        from uni_layer.metrics import MembershipInferenceRisk

        m = MembershipInferenceRisk()
        assert m.name == "membership_inference"
        assert m.category == "security"
        assert m.requires_gradient is True

    def test_compute(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import MembershipInferenceRisk

        m = MembershipInferenceRisk(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert "gradient_entropy" in result
        assert "gradient_snr" in result
        assert "gradient_memorization" in result
        assert "mi_risk_score" in result

    def test_risk_score_range(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import MembershipInferenceRisk

        m = MembershipInferenceRisk(num_batches=3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert result["mi_risk_score"] >= 0
        assert result["gradient_entropy"] >= 0
        assert result["gradient_memorization"] >= 0

    def test_no_params_layer(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import MembershipInferenceRisk

        m = MembershipInferenceRisk(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert result["mi_risk_score"] == 0.0

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "membership_inference" in METRIC_PRIMARY_KEYS


# ──────────────────── AttentionPathTrace ────────────────────


class TestAttentionPathTrace:
    def test_import(self):
        from uni_layer.metrics import AttentionPathTrace

        assert AttentionPathTrace is not None

    def test_init(self):
        from uni_layer.metrics import AttentionPathTrace

        m = AttentionPathTrace()
        assert m.name == "attention_path_trace"
        assert m.category == "security"

    def test_non_attention_layer(self, simple_model, sample_data):
        """Non-attention layers should use proxy or return None."""
        from uni_layer.metrics import AttentionPathTrace

        m = AttentionPathTrace(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        # Linear layer has no attention, should return None values
        assert result["attention_manipulability"] is None

    def test_attention_layer(self, transformer_model, sample_data):
        """Transformer block with attention should return valid metrics."""
        from uni_layer.metrics import AttentionPathTrace

        m = AttentionPathTrace(num_batches=2)
        block = transformer_model.blocks[0]
        result = m.compute(
            model=transformer_model,
            layer=block,
            layer_name="blocks.0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        # Should have at least concentration and persistence from proxy
        assert result["attention_concentration"] is not None
        assert result["injection_vulnerability"] is not None

    def test_schema(self):
        from uni_layer.core.schema import METRIC_PRIMARY_KEYS

        assert "attention_path_trace" in METRIC_PRIMARY_KEYS
