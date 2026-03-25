"""
Tests for integration bridges (TorchPruning, PEFT, Distillation).
"""

import pytest
import torch.nn as nn

from uni_layer.integrations import (
    DistillationBridge,
    HuggingFacePEFTBridge,
    TorchPruningBridge,
)


@pytest.fixture
def model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def contributions():
    return {
        "0": {"gradient_norm": 1.5, "cka_score": 0.8},
        "2": {"gradient_norm": 0.5, "cka_score": 0.3},
    }


class TestTorchPruningBridge:
    def test_importance_scores(self, model, contributions):
        bridge = TorchPruningBridge(model, contributions)
        scores = bridge.as_importance_scores("gradient_norm")
        assert "0" in scores
        assert "2" in scores
        assert scores["0"] >= scores["2"]

    def test_importance_normalized(self, model, contributions):
        bridge = TorchPruningBridge(model, contributions)
        scores = bridge.as_importance_scores("gradient_norm", normalize=True)
        assert max(scores.values()) == pytest.approx(1.0)
        assert min(scores.values()) == pytest.approx(0.0)

    def test_protected_layers(self, model, contributions):
        bridge = TorchPruningBridge(model, contributions)
        protected = bridge.get_protected_layers("gradient_norm", top_k=1)
        assert len(protected) == 1
        assert protected[0] == "0"

    def test_pruning_ratios(self, model, contributions):
        bridge = TorchPruningBridge(model, contributions)
        ratios = bridge.as_layer_pruning_ratios("gradient_norm", target_sparsity=0.5)
        assert len(ratios) > 0
        for module, ratio in ratios.items():
            assert 0 <= ratio <= 1.0

    def test_missing_metric_raises(self, model, contributions):
        bridge = TorchPruningBridge(model, contributions)
        with pytest.raises(ValueError):
            bridge.as_importance_scores("nonexistent_metric")

    def test_summary(self, model, contributions):
        bridge = TorchPruningBridge(model, contributions)
        s = bridge.summary("gradient_norm")
        assert "Torch-Pruning" in s


class TestHuggingFacePEFTBridge:
    def test_recommend_target_modules(self, model, contributions):
        bridge = HuggingFacePEFTBridge(model, contributions)
        targets = bridge.recommend_target_modules("gradient_norm", top_k=2)
        assert isinstance(targets, list)
        assert len(targets) <= 2

    def test_adaptive_ranks(self, model, contributions):
        bridge = HuggingFacePEFTBridge(model, contributions)
        targets = bridge.recommend_target_modules("gradient_norm")
        ranks = bridge.recommend_adaptive_ranks(
            "gradient_norm",
            base_rank=8,
            max_rank=64,
            target_modules=targets,
        )
        for name, rank in ranks.items():
            assert rank >= 8
            assert rank <= 64
            assert rank % 8 == 0

    def test_lora_config_params(self, model, contributions):
        bridge = HuggingFacePEFTBridge(model, contributions)
        params = bridge.recommend_lora_config_params("gradient_norm")
        assert "target_modules" in params
        assert "r" in params
        assert "lora_alpha" in params

    def test_summary(self, model, contributions):
        bridge = HuggingFacePEFTBridge(model, contributions)
        s = bridge.summary("gradient_norm")
        assert "PEFT" in s


class TestDistillationBridge:
    def test_recommend_layers(self, contributions):
        teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        student = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        bridge = DistillationBridge(teacher, student, contributions)
        layers = bridge.recommend_distillation_layers("gradient_norm", top_k=2)
        assert isinstance(layers, list)
        assert len(layers) <= 2

    def test_layer_pairs(self, contributions):
        teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        student = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        bridge = DistillationBridge(teacher, student, contributions)
        pairs = bridge.recommend_layer_pairs("gradient_norm", top_k=2)
        for t, s in pairs:
            assert isinstance(t, str)
            assert isinstance(s, str)

    def test_layer_weights(self, contributions):
        teacher = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        student = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        bridge = DistillationBridge(teacher, student, contributions)
        weights = bridge.recommend_layer_weights("gradient_norm", normalize=True)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_summary(self, contributions):
        teacher = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        student = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        bridge = DistillationBridge(teacher, student, contributions)
        s = bridge.summary("gradient_norm")
        assert "Distillation" in s
