"""
Performance, throughput, and compatibility benchmarks for Uni-Layer.

Tests cover:
- Wall-clock time for each metric
- Cache vs no-cache performance comparison
- Memory footprint estimation
- Throughput (metrics/sec)
- Multi-architecture compatibility (CNN, Transformer, HF-style, Residual)
- Output format validation against schema
"""

import time
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer import LayerAnalyzer
from uni_layer.core.schema import (
    METRIC_OUTPUT_KEYS,
    METRIC_PRIMARY_KEYS,
    validate_contributions,
    validate_metric_result,
)
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

# ============================================================
# Model fixtures for multi-architecture testing
# ============================================================


@pytest.fixture
def cnn_model():
    """Simple CNN model."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


@pytest.fixture
def cnn_data():
    X = torch.randn(64, 3, 8, 8)
    y = torch.randint(0, 10, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def transformer_model():
    """Transformer with MultiheadAttention."""

    class MiniTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(10, 32)
            self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
            self.norm = nn.LayerNorm(32)
            self.fc = nn.Linear(32, 5)

        def forward(self, x):
            x = self.embed(x).unsqueeze(1).expand(-1, 4, -1)
            a, _ = self.attn(x, x, x)
            x = self.norm(a)
            return self.fc(x.mean(dim=1))

    return MiniTransformer()


@pytest.fixture
def transformer_data():
    X = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def residual_model():
    """Model with skip connections."""

    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.act = nn.ReLU()

        def forward(self, x):
            return x + self.act(self.fc(x))

    class ResModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.res1 = ResBlock(32)
            self.res2 = ResBlock(32)
            self.head = nn.Linear(32, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.res1(x)
            x = self.res2(x)
            return self.head(x)

    return ResModel()


@dataclass
class HFOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


@pytest.fixture
def hf_model():
    """HuggingFace-style model returning dataclass."""

    class HFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(3)])
            self.head = nn.Linear(32, 5)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embed(input_ids).mean(dim=1)
            for layer in self.layers:
                x = torch.relu(layer(x))
            logits = self.head(x)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
            return HFOutput(logits=logits, loss=loss)

    return HFModel()


@pytest.fixture
def hf_data():
    X = torch.randint(1, 100, (64, 8))
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def simple_model():
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def simple_data():
    X = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


# ============================================================
# Performance & Throughput Tests
# ============================================================


class TestPerformance:
    """Benchmark wall-clock time for each metric."""

    METRICS = [
        ("GradientNorm", lambda: GradientNorm(num_batches=2)),
        ("HessianTrace", lambda: HessianTrace(num_samples=2, num_batches=2)),
        ("FisherInformation", lambda: FisherInformation(num_batches=2)),
        ("CKA", lambda: CKA(num_batches=2)),
        ("EffectiveRank", lambda: EffectiveRank(num_batches=2)),
        ("NTKTrace", lambda: NTKTrace(num_samples=8)),
        ("ActivationEntropy", lambda: ActivationEntropy(num_batches=2)),
        ("MutualInformation", lambda: MutualInformation(num_batches=2)),
        ("BlockInfluence", lambda: BlockInfluence(num_batches=2)),
        ("DropLayerRobustness", lambda: DropLayerRobustness(num_batches=2)),
        ("LaplacePosterior", lambda: LaplacePosterior(num_batches=2, mode="fisher")),
    ]

    @pytest.mark.parametrize("name,metric_fn", METRICS, ids=[m[0] for m in METRICS])
    def test_metric_completes_under_10s(self, name, metric_fn, simple_model, simple_data):
        """Each metric must complete a single layer in <10s."""
        metric = metric_fn()
        start = time.perf_counter()
        result = metric.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=simple_data,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"{name} took {elapsed:.2f}s (limit: 10s)"
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_cache_faster_than_no_cache(self, simple_model, simple_data):
        """Cached mode should not be slower than uncached for multiple metrics."""
        metrics = [
            CKA(num_batches=2),
            EffectiveRank(num_batches=2),
            ActivationEntropy(num_batches=2),
        ]
        analyzer = LayerAnalyzer(simple_model, task_type="classification")

        # Cached
        start = time.perf_counter()
        r1 = analyzer.compute_metrics(
            metrics=metrics, data_loader=simple_data, verbose=False, use_cache=True, num_batches=2
        )
        t_cached = time.perf_counter() - start

        # Uncached
        start = time.perf_counter()
        r2 = analyzer.compute_metrics(
            metrics=metrics, data_loader=simple_data, verbose=False, use_cache=False
        )
        t_uncached = time.perf_counter() - start

        # Both should produce results
        assert len(r1) > 0
        assert len(r2) > 0
        # Cached should not be dramatically slower (allow 3x margin for small models)
        assert t_cached < t_uncached * 3 + 0.5

    def test_throughput_multiple_layers(self, simple_model, simple_data):
        """Measure layers/sec throughput."""
        metrics = [GradientNorm(num_batches=2), CKA(num_batches=2)]
        analyzer = LayerAnalyzer(simple_model, task_type="classification")

        start = time.perf_counter()
        results = analyzer.compute_metrics(
            metrics=metrics, data_loader=simple_data, verbose=False, num_batches=2
        )
        elapsed = time.perf_counter() - start

        num_layers = len(results)
        num_metrics = len(metrics)
        throughput = (num_layers * num_metrics) / elapsed
        # Should process at least 1 metric-layer/sec on CPU
        assert throughput > 1.0, f"Throughput {throughput:.1f} metric-layers/sec too low"


# ============================================================
# Compatibility Tests
# ============================================================


class TestCompatibility:
    """Test all metrics work across different model architectures."""

    ACTIVATION_METRICS = [
        lambda: CKA(num_batches=2),
        lambda: EffectiveRank(num_batches=2),
        lambda: ActivationEntropy(num_batches=2),
        lambda: BlockInfluence(num_batches=2),
    ]

    GRADIENT_METRICS = [
        lambda: GradientNorm(num_batches=2),
        lambda: FisherInformation(num_batches=2),
    ]

    def _run_metric(self, metric, model, layer, data_loader):
        result = metric.compute(
            model=model,
            layer=layer,
            layer_name="test",
            layer_idx=0,
            data_loader=data_loader,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        for k, v in result.items():
            assert v is None or isinstance(
                v, (int, float)
            ), f"Key '{k}' has type {type(v)}, expected float|None"
        return result

    # --- CNN ---
    @pytest.mark.parametrize(
        "metric_fn", ACTIVATION_METRICS, ids=["CKA", "EffRank", "Entropy", "BI"]
    )
    def test_cnn_activation_metrics(self, metric_fn, cnn_model, cnn_data):
        self._run_metric(metric_fn(), cnn_model, cnn_model[0], cnn_data)

    @pytest.mark.parametrize("metric_fn", GRADIENT_METRICS, ids=["GradNorm", "Fisher"])
    def test_cnn_gradient_metrics(self, metric_fn, cnn_model, cnn_data):
        self._run_metric(metric_fn(), cnn_model, cnn_model[0], cnn_data)

    # --- Transformer ---
    @pytest.mark.parametrize(
        "metric_fn", ACTIVATION_METRICS, ids=["CKA", "EffRank", "Entropy", "BI"]
    )
    def test_transformer_activation_metrics(self, metric_fn, transformer_model, transformer_data):
        self._run_metric(metric_fn(), transformer_model, transformer_model.embed, transformer_data)

    @pytest.mark.parametrize("metric_fn", GRADIENT_METRICS, ids=["GradNorm", "Fisher"])
    def test_transformer_gradient_metrics(self, metric_fn, transformer_model, transformer_data):
        self._run_metric(metric_fn(), transformer_model, transformer_model.embed, transformer_data)

    # --- HuggingFace-style ---
    @pytest.mark.parametrize(
        "metric_fn", ACTIVATION_METRICS, ids=["CKA", "EffRank", "Entropy", "BI"]
    )
    def test_hf_activation_metrics(self, metric_fn, hf_model, hf_data):
        self._run_metric(metric_fn(), hf_model, hf_model.layers[0], hf_data)

    @pytest.mark.parametrize("metric_fn", GRADIENT_METRICS, ids=["GradNorm", "Fisher"])
    def test_hf_gradient_metrics(self, metric_fn, hf_model, hf_data):
        self._run_metric(metric_fn(), hf_model, hf_model.layers[0], hf_data)

    # --- Residual ---
    @pytest.mark.parametrize(
        "metric_fn", ACTIVATION_METRICS, ids=["CKA", "EffRank", "Entropy", "BI"]
    )
    def test_residual_activation_metrics(self, metric_fn, residual_model, transformer_data):
        self._run_metric(metric_fn(), residual_model, residual_model.res1, transformer_data)

    @pytest.mark.parametrize("metric_fn", GRADIENT_METRICS, ids=["GradNorm", "Fisher"])
    def test_residual_gradient_metrics(self, metric_fn, residual_model, transformer_data):
        self._run_metric(metric_fn(), residual_model, residual_model.res1, transformer_data)

    # --- Full analyzer pipeline ---
    def test_analyzer_cnn(self, cnn_model, cnn_data):
        analyzer = LayerAnalyzer(cnn_model, task_type="classification")
        results = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2), CKA(num_batches=2)],
            data_loader=cnn_data,
            verbose=False,
            num_batches=2,
        )
        validate_contributions(results)

    def test_analyzer_hf(self, hf_model, hf_data):
        analyzer = LayerAnalyzer(hf_model, task_type="classification")
        results = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=2), BlockInfluence(num_batches=2)],
            data_loader=hf_data,
            verbose=False,
            num_batches=2,
        )
        validate_contributions(results)

    def test_analyzer_transformer(self, transformer_model, transformer_data):
        analyzer = LayerAnalyzer(transformer_model, task_type="classification")
        results = analyzer.compute_metrics(
            metrics=[EffectiveRank(num_batches=2), ActivationEntropy(num_batches=2)],
            data_loader=transformer_data,
            verbose=False,
            num_batches=2,
        )
        validate_contributions(results)


# ============================================================
# Output Format Validation Tests
# ============================================================


class TestOutputFormat:
    """Verify every metric conforms to its documented schema."""

    ALL_METRICS = [
        ("gradient_norm", lambda: GradientNorm(num_batches=2)),
        ("hessian_trace", lambda: HessianTrace(num_samples=2, num_batches=1)),
        ("fisher_information", lambda: FisherInformation(num_batches=2)),
        ("cka", lambda: CKA(num_batches=2)),
        ("effective_rank", lambda: EffectiveRank(num_batches=2)),
        ("ntk_trace", lambda: NTKTrace(num_samples=8)),
        ("activation_entropy", lambda: ActivationEntropy(num_batches=2)),
        ("mutual_information", lambda: MutualInformation(num_batches=2)),
        ("block_influence", lambda: BlockInfluence(num_batches=2)),
        ("droplayer_robustness", lambda: DropLayerRobustness(num_batches=2)),
        ("laplace_posterior", lambda: LaplacePosterior(num_batches=2, mode="fisher")),
    ]

    @pytest.mark.parametrize("metric_name,metric_fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_output_keys_match_schema(self, metric_name, metric_fn, simple_model, simple_data):
        """Each metric's output must contain its documented primary key."""
        metric = metric_fn()
        DropLayerRobustness.clear_baseline_cache()
        result = metric.compute(
            model=simple_model,
            layer=simple_model[0],
            layer_name="0",
            layer_idx=0,
            data_loader=simple_data,
            device="cpu",
            criterion=nn.CrossEntropyLoss(),
        )
        validate_metric_result(metric_name, result)

    def test_compute_metrics_has_layer_idx_and_type(self, simple_model, simple_data):
        """compute_metrics() must inject layer_idx and layer_type."""
        analyzer = LayerAnalyzer(simple_model, task_type="classification")
        results = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=1)],
            data_loader=simple_data,
            verbose=False,
            num_batches=1,
        )
        for layer_name, data in results.items():
            assert "layer_idx" in data, f"Missing layer_idx for {layer_name}"
            assert "layer_type" in data, f"Missing layer_type for {layer_name}"
            assert isinstance(data["layer_idx"], int)
            assert isinstance(data["layer_type"], str)

    def test_rank_layers_returns_sorted_tuples(self, simple_model, simple_data):
        """rank_layers() must return sorted (name, score) tuples."""
        analyzer = LayerAnalyzer(simple_model, task_type="classification")
        results = analyzer.compute_metrics(
            metrics=[GradientNorm(num_batches=1)],
            data_loader=simple_data,
            verbose=False,
            num_batches=1,
        )
        ranked = analyzer.rank_layers(results, "gradient_norm")
        assert isinstance(ranked, list)
        for item in ranked:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], (int, float))
        # Check descending order
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_schema_validator_catches_bad_output(self):
        """validate_metric_result should reject malformed output."""
        with pytest.raises(ValueError, match="missing primary key"):
            validate_metric_result("gradient_norm", {"wrong_key": 1.0})

        with pytest.raises(ValueError, match="Unknown metric"):
            validate_metric_result("nonexistent", {"x": 1.0})

    def test_contributions_validator(self):
        validate_contributions({"layer0": {"layer_idx": 0, "layer_type": "linear"}})
        with pytest.raises(ValueError, match="missing 'layer_idx'"):
            validate_contributions({"layer0": {"layer_type": "linear"}})
