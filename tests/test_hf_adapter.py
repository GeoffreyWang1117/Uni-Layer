"""
Tests for HuggingFace model adaptation and Block Influence metric.
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer.utils.model_adapter import compute_loss, extract_logits, model_forward

# ---------- Mock HuggingFace output types ----------


@dataclass
class MockModelOutput:
    """Simulates HuggingFace BaseModelOutputWithPooling."""

    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class HFStyleModel(nn.Module):
    """Model that returns dict-like output (simulating HuggingFace)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layer1 = nn.Linear(32, 32)
        self.layer2 = nn.Linear(32, 5)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        x = self.layer1(x.mean(dim=1))
        logits = self.layer2(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return MockModelOutput(logits=logits, loss=loss)


class HFStyleModelNoLogits(nn.Module):
    """Model returning last_hidden_state only (like a base BERT)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layer = nn.Linear(32, 32)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        return MockModelOutput(last_hidden_state=self.layer(x))


# ---------- Tests for extract_logits ----------


class TestExtractLogits:
    def test_tensor_passthrough(self):
        t = torch.randn(4, 5)
        assert extract_logits(t) is t

    def test_dataclass_with_logits(self):
        logits = torch.randn(4, 5)
        out = MockModelOutput(logits=logits)
        assert extract_logits(out) is logits

    def test_dataclass_with_last_hidden_state(self):
        hidden = torch.randn(4, 8, 32)
        out = MockModelOutput(last_hidden_state=hidden)
        assert extract_logits(out) is hidden

    def test_dict_output(self):
        logits = torch.randn(4, 5)
        out = {"logits": logits, "hidden_states": None}
        assert extract_logits(out) is logits

    def test_tuple_output(self):
        logits = torch.randn(4, 5)
        out = (logits, torch.randn(4, 32))
        assert extract_logits(out) is logits

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Cannot extract logits"):
            extract_logits("not a tensor")


# ---------- Tests for compute_loss ----------


class TestComputeLoss:
    def test_with_internal_loss(self):
        loss = torch.tensor(0.5)
        out = MockModelOutput(logits=torch.randn(4, 5), loss=loss)
        result = compute_loss(out, None, None)
        assert result is loss

    def test_with_criterion(self):
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        out = MockModelOutput(logits=logits)
        result = compute_loss(out, targets, nn.CrossEntropyLoss())
        assert result.dim() == 0

    def test_fallback_mean(self):
        logits = torch.randn(4, 5)
        result = compute_loss(logits, None, None)
        assert result.item() == pytest.approx(logits.mean().item())


# ---------- Tests for model_forward ----------


class TestModelForward:
    def test_standard_model(self):
        model = nn.Sequential(nn.Linear(10, 5))
        x = torch.randn(4, 10)
        out = model_forward(model, x)
        assert isinstance(out, torch.Tensor)

    def test_hf_model_attention_mask(self):
        model = HFStyleModel()
        input_ids = torch.randint(1, 100, (4, 8))
        out = model_forward(model, input_ids)
        assert hasattr(out, "logits")
        assert out.logits.shape == (4, 5)

    def test_hf_model_with_labels(self):
        model = HFStyleModel()
        input_ids = torch.randint(1, 100, (4, 8))
        labels = torch.randint(0, 5, (4,))
        out = model_forward(model, input_ids, labels)
        assert out.loss is not None


# ---------- Tests for metrics with HF-style models ----------


class TestMetricsWithHFModel:
    @pytest.fixture
    def hf_model_and_data(self):
        model = HFStyleModel()
        input_ids = torch.randint(1, 100, (64, 8))
        labels = torch.randint(0, 5, (64,))
        loader = DataLoader(TensorDataset(input_ids, labels), batch_size=16)
        return model, loader

    def test_gradient_norm_hf(self, hf_model_and_data):
        from uni_layer.metrics import GradientNorm

        model, loader = hf_model_and_data
        m = GradientNorm(num_batches=2)
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        assert result["gradient_norm"] >= 0

    def test_cka_hf(self, hf_model_and_data):
        from uni_layer.metrics import CKA

        model, loader = hf_model_and_data
        m = CKA(num_batches=2)
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
        )
        assert "cka_score" in result

    def test_effective_rank_hf(self, hf_model_and_data):
        from uni_layer.metrics import EffectiveRank

        model, loader = hf_model_and_data
        m = EffectiveRank(num_batches=2)
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
        )
        assert "effective_rank" in result

    def test_fisher_information_hf(self, hf_model_and_data):
        from uni_layer.metrics import FisherInformation

        model, loader = hf_model_and_data
        m = FisherInformation(num_batches=2)
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        assert result["fisher_information"] >= 0

    def test_droplayer_hf(self, hf_model_and_data):
        from uni_layer.metrics import DropLayerRobustness

        model, loader = hf_model_and_data
        DropLayerRobustness.clear_baseline_cache()
        m = DropLayerRobustness(num_batches=2)
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        assert "droplayer_loss_increase" in result

    def test_activation_entropy_hf(self, hf_model_and_data):
        from uni_layer.metrics import ActivationEntropy

        model, loader = hf_model_and_data
        m = ActivationEntropy(num_batches=2)
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
        )
        assert result["activation_entropy"] >= 0

    def test_hessian_trace_hf(self, hf_model_and_data):
        from uni_layer.metrics import HessianTrace

        model, loader = hf_model_and_data
        m = HessianTrace(num_samples=2, num_batches=1)
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        assert "hessian_trace" in result

    def test_laplace_posterior_hf(self, hf_model_and_data):
        from uni_layer.metrics import LaplacePosterior

        model, loader = hf_model_and_data
        m = LaplacePosterior(num_batches=2, mode="fisher")
        result = m.compute(
            model=model,
            layer=model.layer1,
            layer_name="layer1",
            layer_idx=0,
            data_loader=loader,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        assert result["laplace_posterior"] >= 0


# ---------- Tests for Block Influence metric ----------


class TestBlockInfluence:
    @pytest.fixture
    def residual_model(self):
        """Model with residual connection — BI should be low."""

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(20, 20)
                # Initialize near-zero so output ≈ input (residual)
                nn.init.zeros_(self.linear.weight)
                nn.init.zeros_(self.linear.bias)

            def forward(self, x):
                return x + self.linear(x)  # residual

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.res = ResBlock()
                self.fc2 = nn.Linear(20, 5)

            def forward(self, x):
                return self.fc2(self.res(torch.relu(self.fc1(x))))

        return Model()

    @pytest.fixture
    def sample_data(self):
        X = torch.randn(64, 10)
        y = torch.randint(0, 5, (64,))
        return DataLoader(TensorDataset(X, y), batch_size=16)

    def test_initialization(self):
        from uni_layer.metrics import BlockInfluence

        m = BlockInfluence()
        assert m.name == "block_influence"
        assert m.category == "representation"
        assert m.requires_gradient is False

    def test_compute(self, sample_data):
        from uni_layer.metrics import BlockInfluence

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        m = BlockInfluence(num_batches=2)
        result = m.compute(
            model=model,
            layer=model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=sample_data,
            device="cpu",
        )
        assert "block_influence" in result
        assert "block_similarity" in result
        assert 0 <= result["block_influence"] <= 2.0  # BI in [0, 2] for cosine
        assert -1 <= result["block_similarity"] <= 1.0

    def test_residual_low_bi(self, residual_model, sample_data):
        """A near-identity residual block should have low BI (high similarity)."""
        from uni_layer.metrics import BlockInfluence

        m = BlockInfluence(num_batches=2)
        result = m.compute(
            model=residual_model,
            layer=residual_model.res,
            layer_name="res",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )
        # With zero-initialized weights, output ≈ input
        assert result["block_influence"] < 0.1
        assert result["block_similarity"] > 0.9

    def test_transforming_layer_high_bi(self, sample_data):
        """A layer that significantly transforms input should have high BI."""
        from uni_layer.metrics import BlockInfluence

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        m = BlockInfluence(num_batches=2)
        # ReLU changes the representation significantly
        result = m.compute(
            model=model,
            layer=model[1],  # ReLU
            layer_name="1",
            layer_idx=1,
            data_loader=sample_data,
            device="cpu",
        )
        assert result["block_influence"] > 0.0

    def test_importable(self):
        from uni_layer.metrics import BlockInfluence

        assert BlockInfluence is not None

    def test_all_13_metrics_importable(self):
        """Verify all 13 metrics (12 + BlockInfluence) can be imported."""
        from uni_layer.metrics import (
            CKA,
            ActivationEntropy,
            AttentionFlow,
            BlockInfluence,
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

        assert (
            len(
                [
                    GradientNorm,
                    HessianTrace,
                    FisherInformation,
                    CKA,
                    EffectiveRank,
                    NTKTrace,
                    MutualInformation,
                    ActivationEntropy,
                    JacobianRank,
                    BlockInfluence,
                    DropLayerRobustness,
                    LaplacePosterior,
                    AttentionFlow,
                ]
            )
            == 13
        )
