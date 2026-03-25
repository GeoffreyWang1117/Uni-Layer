"""
Comprehensive tests for all 12 layer contribution metrics.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def simple_model():
    """Simple 3-layer model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 5),
    )
    return model


@pytest.fixture
def sample_data():
    """Classification dataset with 100 samples."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Optimization metrics
# ---------------------------------------------------------------------------


class TestHessianTrace:
    def test_initialization(self):
        from uni_layer.metrics import HessianTrace

        m = HessianTrace()
        assert m.name == "hessian_trace"
        assert m.category == "optimization"
        assert m.requires_gradient is True

    def test_compute(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import HessianTrace

        m = HessianTrace(num_samples=2, num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert "hessian_trace" in result
        assert "hessian_trace_std" in result
        assert isinstance(result["hessian_trace"], float)

    def test_no_params_layer(self, simple_model, sample_data):
        from uni_layer.metrics import HessianTrace

        m = HessianTrace(num_samples=2, num_batches=1)
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],  # ReLU has no params
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["hessian_trace"] == 0.0


class TestFisherInformation:
    def test_initialization(self):
        from uni_layer.metrics import FisherInformation

        m = FisherInformation()
        assert m.name == "fisher_information"
        assert m.category == "optimization"

    def test_compute_empirical(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import FisherInformation

        m = FisherInformation(num_batches=2, empirical=True)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert "fisher_information" in result
        assert "fisher_mean" in result
        assert result["fisher_information"] >= 0

    def test_compute_true_fisher(self, simple_model, sample_data):
        from uni_layer.metrics import FisherInformation

        m = FisherInformation(num_batches=2, empirical=False)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["fisher_information"] >= 0


class TestGradientNorm:
    def test_compute_all_norms(self, simple_model, sample_data):
        from uni_layer.metrics import GradientNorm

        for norm_type in ["l1", "l2", "linf"]:
            m = GradientNorm(norm_type=norm_type, num_batches=1)
            result = m.compute(
                model=simple_model,
                layer=simple_model[0],
                layer_name="0",
                layer_idx=0,
                data_loader=sample_data,
                device="cpu",
            )
            assert result["gradient_norm"] >= 0


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------


class TestNTKTrace:
    def test_initialization(self):
        from uni_layer.metrics import NTKTrace

        m = NTKTrace()
        assert m.name == "ntk_trace"
        assert m.category == "spectral"
        assert m.requires_gradient is True

    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import NTKTrace

        m = NTKTrace(num_samples=8)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "ntk_trace" in result
        assert "ntk_trace_per_param" in result
        assert result["ntk_trace"] >= 0

    def test_no_params(self, simple_model, sample_data):
        from uni_layer.metrics import NTKTrace

        m = NTKTrace(num_samples=4)
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],  # ReLU
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["ntk_trace"] == 0.0


class TestCKA:
    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import CKA

        m = CKA(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "cka_score" in result
        assert 0 <= result["cka_score"] <= 1


class TestEffectiveRank:
    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import EffectiveRank

        m = EffectiveRank(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "effective_rank" in result
        assert "stable_rank" in result
        assert "rank_ratio" in result
        assert result["effective_rank"] >= 0


# ---------------------------------------------------------------------------
# Information theory metrics
# ---------------------------------------------------------------------------


class TestMutualInformation:
    def test_initialization(self):
        from uni_layer.metrics import MutualInformation

        m = MutualInformation()
        assert m.name == "mutual_information"
        assert m.category == "information_theory"
        assert m.requires_gradient is False

    def test_compute_classification(self, simple_model, sample_data):
        from uni_layer.metrics import MutualInformation

        m = MutualInformation(num_batches=2, task_type="classification")
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "mutual_information" in result
        assert result["mutual_information"] >= 0

    def test_compute_regression(self):
        from uni_layer.metrics import MutualInformation

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
        X = torch.randn(100, 10)
        y = torch.randn(100)
        loader = DataLoader(TensorDataset(X, y), batch_size=16)

        m = MutualInformation(num_batches=2, task_type="regression")
        result = m.compute(
            model=model,
            layer=model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
        )
        assert "mutual_information" in result


class TestActivationEntropy:
    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import ActivationEntropy

        m = ActivationEntropy(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "activation_entropy" in result
        assert result["activation_entropy"] >= 0
        assert "activation_sparsity" in result
        assert 0 <= result["activation_sparsity"] <= 1


# ---------------------------------------------------------------------------
# Representation metrics
# ---------------------------------------------------------------------------


class TestJacobianRank:
    def test_initialization(self):
        from uni_layer.metrics import JacobianRank

        m = JacobianRank()
        assert m.name == "jacobian_rank"
        assert m.category == "representation"

    def test_compute(self, simple_model, sample_data):
        from uni_layer.metrics import JacobianRank

        m = JacobianRank(num_samples=8, rank_threshold=1e-3)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "jacobian_rank" in result
        assert "jacobian_rank_ratio" in result
        assert "jacobian_condition" in result
        assert "jacobian_max_sv" in result


# ---------------------------------------------------------------------------
# Robustness metrics
# ---------------------------------------------------------------------------


class TestDropLayerRobustness:
    def test_compute(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import DropLayerRobustness

        m = DropLayerRobustness(num_batches=2)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert "droplayer_loss_increase" in result
        assert isinstance(result["droplayer_loss_increase"], float)


# ---------------------------------------------------------------------------
# Bayesian metrics
# ---------------------------------------------------------------------------


class TestLaplacePosterior:
    def test_initialization(self):
        from uni_layer.metrics import LaplacePosterior

        m = LaplacePosterior()
        assert m.name == "laplace_posterior"
        assert m.category == "bayesian"

    def test_fisher_mode(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import LaplacePosterior

        m = LaplacePosterior(num_batches=2, mode="fisher")
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert "laplace_posterior" in result
        assert result["laplace_posterior"] >= 0

    def test_hutchinson_mode(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import LaplacePosterior

        m = LaplacePosterior(num_samples=2, num_batches=2, mode="hutchinson")
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert "laplace_posterior" in result
        assert "laplace_posterior_std" in result

    def test_no_params(self, simple_model, sample_data, criterion):
        from uni_layer.metrics import LaplacePosterior

        m = LaplacePosterior(num_batches=1, mode="fisher")
        result = m.compute(
            model=simple_model,
            layer=simple_model[1],  # ReLU
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
            criterion=criterion,
        )
        assert result["laplace_posterior"] == 0.0


# ---------------------------------------------------------------------------
# Architecture-specific metrics
# ---------------------------------------------------------------------------


class TestAttentionFlow:
    def test_initialization(self):
        from uni_layer.metrics import AttentionFlow

        m = AttentionFlow()
        assert m.name == "attention_flow"
        assert m.category == "architecture_specific"
        assert m.requires_gradient is False

    def test_non_attention_layer(self, simple_model, sample_data):
        from uni_layer.metrics import AttentionFlow

        m = AttentionFlow(num_batches=1)
        result = m.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        # Non-attention layer returns None values
        assert result["attention_entropy"] is None

    def test_with_multihead_attention(self):
        from uni_layer.metrics import AttentionFlow

        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(10, 32)
                self.attn = nn.MultiheadAttention(32, num_heads=4, batch_first=True)
                self.fc = nn.Linear(32, 5)

            def forward(self, x):
                x = self.embed(x).unsqueeze(1).expand(-1, 4, -1)  # (B, seq=4, 32)
                x, _ = self.attn(x, x, x, need_weights=True)
                return self.fc(x.mean(dim=1))

        model = SimpleTransformer()
        X = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)

        m = AttentionFlow(num_batches=2)
        result = m.compute(
            model=model,
            layer=model.attn,
            layer_name="attn",
            layer_idx=1,
            data_loader=loader,
            device="cpu",
        )
        # Should get actual attention metrics
        assert "attention_entropy" in result
        assert "attention_max_weight" in result

    def test_attention_entropy_computation(self):
        from uni_layer.metrics.architecture_specific.attention_flow import AttentionFlow

        m = AttentionFlow()
        # Uniform attention => max entropy
        uniform = torch.ones(2, 4, 8, 8) / 8.0
        entropy = m._compute_attention_entropy(uniform)
        assert entropy > 0

    def test_head_diversity(self):
        from uni_layer.metrics.architecture_specific.attention_flow import AttentionFlow

        m = AttentionFlow()
        # Identical heads => zero diversity
        attn = torch.randn(2, 4, 8, 8).softmax(dim=-1)
        identical = attn[:, 0:1, :, :].expand(-1, 4, -1, -1)
        div = m._compute_head_diversity(identical)
        assert div < 0.05  # Nearly zero

        # Diverse heads => higher diversity
        diverse = torch.randn(2, 4, 8, 8).softmax(dim=-1)
        div2 = m._compute_head_diversity(diverse)
        assert div2 >= 0


# ---------------------------------------------------------------------------
# Base class tests
# ---------------------------------------------------------------------------


def test_layer_metric_is_abstract():
    from uni_layer.core.base_metric import LayerMetric

    with pytest.raises(TypeError):
        LayerMetric(name="test", category="test")


def test_all_metrics_importable():
    """Verify all 12 metrics can be imported from uni_layer.metrics."""
    from uni_layer.metrics import (
        CKA,
        ActivationEntropy,
        AttentionFlow,
        DropLayerRobustness,
        EffectiveRank,
        FisherInformation,
        GradientNorm,
        HessianTrace,
        JacobianRank,
        LaplacePosterior,
        MutualInformation,
        NTKTrace,
    )

    metrics = [
        GradientNorm,
        HessianTrace,
        FisherInformation,
        CKA,
        EffectiveRank,
        NTKTrace,
        MutualInformation,
        ActivationEntropy,
        JacobianRank,
        DropLayerRobustness,
        LaplacePosterior,
        AttentionFlow,
    ]
    assert len(metrics) == 12
    for cls in metrics:
        obj = cls()
        assert hasattr(obj, "name")
        assert hasattr(obj, "category")
        assert hasattr(obj, "compute")
