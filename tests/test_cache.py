"""
Tests for the caching system (ActivationCache, GradientCache, BatchCache).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from uni_layer.core.cache import ActivationCache, BatchCache, GradientCache


@pytest.fixture
def model_and_layers():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    layers = {"linear_0": model[0], "linear_1": model[2]}
    return model, layers


@pytest.fixture
def loader():
    X = torch.randn(64, 10)
    y = torch.randint(0, 5, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


class TestActivationCache:
    def test_capture(self, model_and_layers, loader):
        model, layers = model_and_layers
        cache = ActivationCache(model, layers)
        cache.capture(loader, num_batches=2, device="cpu")

        assert cache.has("linear_0")
        assert cache.has("linear_1")
        acts = cache.get("linear_0")
        assert len(acts) == 2  # 2 batches

    def test_get_concatenated(self, model_and_layers, loader):
        model, layers = model_and_layers
        cache = ActivationCache(model, layers)
        cache.capture(loader, num_batches=2, device="cpu")

        concat = cache.get_concatenated("linear_0")
        assert concat is not None
        assert concat.shape[0] == 32  # 2 batches x 16

    def test_clear(self, model_and_layers, loader):
        model, layers = model_and_layers
        cache = ActivationCache(model, layers)
        cache.capture(loader, num_batches=1, device="cpu")
        cache.clear()
        assert not cache.has("linear_0")

    def test_missing_layer(self, model_and_layers, loader):
        model, layers = model_and_layers
        cache = ActivationCache(model, layers)
        cache.capture(loader, num_batches=1, device="cpu")
        assert cache.get("nonexistent") == []
        assert cache.get_concatenated("nonexistent") is None


class TestGradientCache:
    def test_capture(self, model_and_layers, loader):
        model, layers = model_and_layers
        cache = GradientCache(model, layers)
        cache.capture(loader, criterion=nn.CrossEntropyLoss(), num_batches=2, device="cpu")

        assert cache.has("linear_0")
        assert cache.has("linear_1")

    def test_clear(self, model_and_layers, loader):
        model, layers = model_and_layers
        cache = GradientCache(model, layers)
        cache.capture(loader, criterion=nn.CrossEntropyLoss(), num_batches=1, device="cpu")
        cache.clear()
        assert not cache.has("linear_0")


class TestBatchCache:
    def test_cache_batches(self, loader):
        cache = BatchCache(loader, num_batches=3, device="cpu")
        assert len(cache) == 3

    def test_iteration(self, loader):
        cache = BatchCache(loader, num_batches=2, device="cpu")
        batches = list(cache)
        assert len(batches) == 2
        assert len(batches[0]) == 2  # (X, y)

    def test_limits_batches(self, loader):
        cache = BatchCache(loader, num_batches=1, device="cpu")
        assert len(cache) == 1
