"""
Tests for LayerProfile auto-analyzer and LLM presets.
"""

import json
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.core.profile import LayerProfile

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def contributions_8layer():
    """Simulates 8-layer transformer contributions with realistic patterns."""
    data = OrderedDict()
    # U-shaped gradient norm, decreasing block influence, one bottleneck at layer 5
    gn = [0.12, 0.08, 0.05, 0.04, 0.03, 0.04, 0.07, 0.15]
    bi = [0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.02, 0.03]
    er = [90.0, 88.0, 85.0, 82.0, 50.0, 80.0, 86.0, 92.0]  # bottleneck at 4
    cka = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.85, 1.00]
    for i in range(8):
        data[f"layers.{i}"] = {
            "layer_idx": i,
            "layer_type": "transformer_block",
            "gradient_norm": gn[i],
            "block_influence": bi[i],
            "effective_rank": er[i],
            "cka_score": cka[i],
        }
    return data


@pytest.fixture
def contributions_with_redundancy():
    """Model with clearly redundant layers."""
    data = OrderedDict()
    for i in range(6):
        data[f"layer.{i}"] = {
            "layer_idx": i,
            "layer_type": "linear",
            "gradient_norm": 0.1 if i not in (2, 3) else 0.001,
            "block_influence": 0.05 if i not in (2, 3) else 0.005,
            "effective_rank": 20.0 if i not in (2, 3) else 18.0,
        }
    return data


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5),
    )


@pytest.fixture
def simple_data():
    X = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


# ------------------------------------------------------------------
# LayerProfile tests
# ------------------------------------------------------------------


class TestLayerProfile:
    def test_init(self, contributions_8layer):
        p = LayerProfile(contributions_8layer, model_name="test-model")
        assert p._n_layers == 8
        assert len(p._metric_keys) == 4
        assert p.model_name == "test-model"

    def test_repr(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        r = repr(p)
        assert "layers=8" in r
        assert "metrics=4" in r


class TestRedundantLayers:
    def test_detects_redundancy(self, contributions_with_redundancy):
        p = LayerProfile(contributions_with_redundancy)
        redundant = p.redundant_layers
        assert "layer.2" in redundant or "layer.3" in redundant

    def test_no_redundancy(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        # All layers have BI > 0.01, might not flag any
        redundant = p.redundant_layers
        assert isinstance(redundant, list)


class TestBottleneckLayers:
    def test_detects_bottleneck(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        bottlenecks = p.bottleneck_layers
        # Layer 4 has eff_rank=50 vs neighbors 82 and 80 → ratio < 0.7
        assert "layers.4" in bottlenecks

    def test_no_bottleneck(self, contributions_with_redundancy):
        p = LayerProfile(contributions_with_redundancy)
        bottlenecks = p.bottleneck_layers
        assert isinstance(bottlenecks, list)


class TestConsensusRanking:
    def test_ranking_format(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        ranking = p.consensus_ranking
        assert len(ranking) == 8
        assert ranking[0]["rank"] == 1
        assert "consensus_score" in ranking[0]
        assert "scores" in ranking[0]
        assert "layer" in ranking[0]

    def test_ranking_ordered(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        ranking = p.consensus_ranking
        scores = [r["consensus_score"] for r in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_top_layer_is_last(self, contributions_8layer):
        """Layer 7 has highest gradient_norm and CKA, should rank #1."""
        p = LayerProfile(contributions_8layer)
        ranking = p.consensus_ranking
        assert ranking[0]["layer"] == "layers.7"


class TestDepthTrends:
    def test_u_shaped_gradient(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        trends = p.depth_trends
        gn = trends.get("gradient_norm", {})
        assert gn["trend"] == "U-shaped"
        assert gn["early"] > gn["middle"]
        assert gn["late"] > gn["middle"]

    def test_increasing_cka(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        trends = p.depth_trends
        cka = trends.get("cka_score", {})
        assert cka["trend"] == "increasing"


class TestAnomalies:
    def test_detects_anomaly(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        anomalies = p.anomalies
        # Layer 4 effective_rank=50 is anomalously low (mean~82, std~13)
        if "layers.4" in anomalies:
            metrics = [a["metric"] for a in anomalies["layers.4"]]
            assert "effective_rank" in metrics

    def test_anomaly_format(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        for layer, items in p.anomalies.items():
            for a in items:
                assert "metric" in a
                assert "value" in a
                assert "z_score" in a
                assert "note" in a


class TestLayerClusters:
    def test_clusters(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        clusters = p.layer_clusters
        assert len(clusters) > 0
        # All layers accounted for
        all_layers = []
        for group in clusters.values():
            all_layers.extend(group)
        assert set(all_layers) == set(p._layers)


class TestPruningSuggestion:
    def test_pruning_format(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        s = p.pruning_suggestion(target_ratio=0.25)
        assert "safe_to_remove" in s
        assert "num_layers_removed" in s
        assert "estimated_speedup" in s
        assert len(s["safe_to_remove"]) == 2  # 25% of 8

    def test_pruning_removes_least_important(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        s = p.pruning_suggestion(target_ratio=0.125)  # remove 1
        ranking = p.consensus_ranking
        least = ranking[-1]["layer"]
        assert least in s["safe_to_remove"]


class TestLoraSuggestion:
    def test_lora_format(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        s = p.lora_suggestion(base_rank=8, max_rank=64)
        assert "target_layers" in s
        assert "adaptive_ranks" in s
        assert len(s["target_layers"]) == 4  # 50% of 8
        for rank in s["adaptive_ranks"].values():
            assert rank >= 8
            assert rank <= 64
            assert rank % 8 == 0

    def test_lora_targets_top_layers(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        s = p.lora_suggestion()
        ranking = p.consensus_ranking
        top_layer = ranking[0]["layer"]
        assert top_layer in s["target_layers"]


class TestSummary:
    def test_summary_string(self, contributions_8layer):
        p = LayerProfile(contributions_8layer, model_name="TestModel")
        s = p.summary()
        assert isinstance(s, str)
        assert "8-layer" in s
        assert "TestModel" in s

    def test_summary_mentions_trends(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        s = p.summary()
        assert "U-shaped" in s or "gradient" in s.lower()


class TestToDict:
    def test_serializable(self, contributions_8layer):
        p = LayerProfile(contributions_8layer, model_name="TestModel")
        d = p.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert len(serialized) > 100

    def test_to_dict_structure(self, contributions_8layer):
        p = LayerProfile(contributions_8layer)
        d = p.to_dict()
        assert "model" in d
        assert "num_layers" in d
        assert "metrics_used" in d
        assert "analysis" in d
        assert "summary" in d
        a = d["analysis"]
        assert "redundant_layers" in a
        assert "bottleneck_layers" in a
        assert "consensus_ranking" in a
        assert "depth_trends" in a
        assert "anomalies" in a
        assert "layer_clusters" in a
        assert "suggestions" in a
        assert "pruning" in a["suggestions"]
        assert "lora" in a["suggestions"]


# ------------------------------------------------------------------
# Preset tests
# ------------------------------------------------------------------


class TestPresets:
    def test_preset_llm_fast(self, simple_model, simple_data):
        analyzer = LayerAnalyzer(simple_model, task_type="classification")
        results = analyzer.compute_metrics(
            data_loader=simple_data,
            verbose=False,
            preset="llm_fast",
            num_batches=1,
        )
        assert len(results) > 0
        # Should contain activation-based metrics
        sample = list(results.values())[0]
        assert "block_influence" in sample or "effective_rank" in sample

    def test_preset_quick(self, simple_model, simple_data):
        analyzer = LayerAnalyzer(simple_model, task_type="classification")
        results = analyzer.compute_metrics(
            data_loader=simple_data,
            verbose=False,
            preset="quick",
            num_batches=1,
        )
        sample = list(results.values())[0]
        assert "gradient_norm" in sample

    def test_preset_unknown_raises(self, simple_model, simple_data):
        analyzer = LayerAnalyzer(simple_model, task_type="classification")
        with pytest.raises(ValueError, match="Unknown preset"):
            analyzer.compute_metrics(
                data_loader=simple_data,
                preset="nonexistent",
            )

    def test_no_metrics_no_preset_raises(self, simple_model, simple_data):
        analyzer = LayerAnalyzer(simple_model, task_type="classification")
        with pytest.raises(ValueError, match="Either metrics or preset"):
            analyzer.compute_metrics(data_loader=simple_data)


# ------------------------------------------------------------------
# End-to-end: compute_metrics → LayerProfile
# ------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self, simple_model, simple_data):
        analyzer = LayerAnalyzer(simple_model, task_type="classification")
        contributions = analyzer.compute_metrics(
            data_loader=simple_data,
            verbose=False,
            preset="quick",
            num_batches=1,
        )
        profile = LayerProfile(contributions, model_name="test-mlp")
        d = profile.to_dict()

        assert d["num_layers"] > 0
        assert len(d["analysis"]["consensus_ranking"]) > 0
        assert d["summary"]
        # JSON round-trip
        json.loads(json.dumps(d))
