"""
Tests for CKA Similarity Matrix (v0.4.0 Feature 1).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def simple_model():
    """Simple 4-layer model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 15),
        nn.ReLU(),
        nn.Linear(15, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
    )
    return model


@pytest.fixture
def sample_data():
    """Sample dataset."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


class TestCKASimilarity:
    def test_import(self):
        from uni_layer.core.cka_similarity import CKASimilarity

        assert CKASimilarity is not None

    def test_top_level_import(self):
        from uni_layer import CKASimilarity

        assert CKASimilarity is not None

    def test_init(self, simple_model):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        assert len(cka.layer_names) > 0

    def test_compute_returns_matrix(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=3, verbose=False)

        n = len(names)
        assert matrix.shape == (n, n)
        assert len(names) == n

    def test_matrix_is_symmetric(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, _ = cka.compute(sample_data, num_batches=3, verbose=False)

        np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=10)

    def test_diagonal_is_one(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, _ = cka.compute(sample_data, num_batches=3, verbose=False)

        np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(matrix.shape[0]), decimal=5)

    def test_values_in_valid_range(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, _ = cka.compute(sample_data, num_batches=3, verbose=False)

        # CKA values should be in [-1, 1] (typically [0, 1] for linear kernel)
        assert matrix.min() >= -1.0 - 1e-6
        assert matrix.max() <= 1.0 + 1e-6

    def test_most_similar_pairs(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=3, verbose=False)

        pairs = cka.most_similar_pairs(matrix, names, top_k=3)
        assert len(pairs) <= 3
        for name_i, name_j, score in pairs:
            assert name_i in names
            assert name_j in names
            assert isinstance(score, (float, np.floating))

        # Should be sorted descending
        scores = [s for _, _, s in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_most_distinct_pairs(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=3, verbose=False)

        pairs = cka.most_distinct_pairs(matrix, names, top_k=3)
        assert len(pairs) <= 3

        # Should be sorted ascending
        scores = [s for _, _, s in pairs]
        assert scores == sorted(scores)

    def test_redundant_layers(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=3, verbose=False)

        # With high threshold, may return empty list
        redundant = cka.redundant_layers(matrix, names, threshold=0.99)
        assert isinstance(redundant, list)

        # With low threshold, should return more pairs
        redundant_low = cka.redundant_layers(matrix, names, threshold=0.0)
        assert len(redundant_low) >= len(redundant)

    def test_layer_uniqueness(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=3, verbose=False)

        uniqueness = cka.layer_uniqueness(matrix, names)
        assert len(uniqueness) == len(names)
        for name in names:
            assert name in uniqueness
            assert isinstance(uniqueness[name], float)

    def test_to_dict(self, simple_model, sample_data):
        from uni_layer.core.cka_similarity import CKASimilarity

        cka = CKASimilarity(simple_model, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=3, verbose=False)

        result = cka.to_dict(matrix, names)
        assert "matrix" in result
        assert "layer_names" in result
        assert "most_similar" in result
        assert "most_distinct" in result
        assert "redundant_pairs" in result
        assert "layer_uniqueness" in result
        assert len(result["matrix"]) == len(names)

    def test_custom_layers(self, simple_model, sample_data):
        """Test with manually specified layer subset."""
        from collections import OrderedDict

        from uni_layer.core.cka_similarity import CKASimilarity

        layers = OrderedDict(
            [
                ("layer_0", simple_model[0]),
                ("layer_2", simple_model[2]),
                ("layer_4", simple_model[4]),
            ]
        )
        cka = CKASimilarity(simple_model, layers=layers, device="cpu")
        matrix, names = cka.compute(sample_data, num_batches=3, verbose=False)

        assert matrix.shape == (3, 3)
        assert names == ["layer_0", "layer_2", "layer_4"]


class TestLayerAnalyzerCKAMatrix:
    def test_compute_cka_matrix(self, simple_model, sample_data):
        from uni_layer.core.analyzer import LayerAnalyzer

        analyzer = LayerAnalyzer(simple_model, task_type="classification", device="cpu")
        matrix, names = analyzer.compute_cka_matrix(
            data_loader=sample_data, num_batches=3, verbose=False
        )

        assert isinstance(matrix, np.ndarray)
        assert isinstance(names, list)
        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.shape[0] == len(names)
