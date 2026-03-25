"""Tests for fast math utilities (randomized SVD, minibatch CKA, etc.)"""

import torch
import numpy as np
import pytest

from uni_layer.utils.fast_math import (
    randomized_svd,
    effective_rank_from_svd,
    minibatch_cka,
)


class TestRandomizedSVD:
    """Test randomized SVD correctness and performance."""

    def test_basic_correctness(self):
        """Randomized SVD should approximate full SVD singular values."""
        torch.manual_seed(42)
        X = torch.randn(100, 50)
        _, S_full, _ = torch.linalg.svd(X, full_matrices=False)

        _, S_rand, _ = randomized_svd(X, n_components=20)

        # Top singular values should be close
        assert S_rand.shape[0] == 20
        torch.testing.assert_close(S_rand, S_full[:20], atol=0.5, rtol=0.1)

    def test_small_matrix_uses_full_svd(self):
        """Small matrices should fall back to full SVD."""
        X = torch.randn(10, 5)
        U, S, Vt = randomized_svd(X, n_components=3)
        assert S.shape[0] == 3

    def test_tall_matrix(self):
        """Should work with tall matrices (N >> D)."""
        X = torch.randn(500, 20)
        U, S, Vt = randomized_svd(X, n_components=10)
        assert U.shape == (500, 10)
        assert S.shape == (10,)
        assert Vt.shape == (10, 20)

    def test_wide_matrix(self):
        """Should work with wide matrices (D >> N)."""
        X = torch.randn(20, 500)
        U, S, Vt = randomized_svd(X, n_components=10)
        assert U.shape == (20, 10)
        assert S.shape == (10,)
        assert Vt.shape == (10, 500)

    def test_singular_values_sorted(self):
        """Singular values should be in descending order."""
        X = torch.randn(100, 80)
        _, S, _ = randomized_svd(X, n_components=30)
        for i in range(len(S) - 1):
            assert S[i] >= S[i + 1]

    def test_reconstruction_quality(self):
        """Low-rank reconstruction should be reasonably close."""
        torch.manual_seed(0)
        # Create low-rank matrix + noise
        A = torch.randn(100, 5)
        B = torch.randn(5, 50)
        X = A @ B + 0.01 * torch.randn(100, 50)

        U, S, Vt = randomized_svd(X, n_components=5)
        X_approx = U * S.unsqueeze(0) @ Vt

        rel_error = (X - X_approx).norm() / X.norm()
        assert rel_error < 0.05  # <5% reconstruction error for rank-5 matrix


class TestEffectiveRankFromSVD:
    """Test effective rank computation from singular values."""

    def test_uniform_singular_values(self):
        """Uniform singular values should give maximum effective rank."""
        S = torch.ones(10)
        rank = effective_rank_from_svd(S)
        assert abs(rank - 10.0) < 0.1  # Should be close to 10

    def test_single_dominant_sv(self):
        """One dominant singular value should give rank ~1."""
        S = torch.tensor([100.0, 0.01, 0.01, 0.01])
        rank = effective_rank_from_svd(S)
        assert rank < 2.0  # Should be close to 1

    def test_zero_singular_values(self):
        """All-zero singular values should return 0."""
        S = torch.zeros(5)
        rank = effective_rank_from_svd(S)
        assert rank == 0.0

    def test_rank_between_1_and_n(self):
        """Effective rank should always be between 0 and N."""
        S = torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1])
        rank = effective_rank_from_svd(S)
        assert 1.0 <= rank <= 5.0


class TestMinibatchCKA:
    """Test minibatch CKA correctness."""

    def test_identical_representations(self):
        """CKA of identical representations should be ~1."""
        X = torch.randn(100, 32)
        cka = minibatch_cka(X, X, batch_size=50)
        assert cka > 0.9  # Should be close to 1

    def test_orthogonal_representations(self):
        """CKA of uncorrelated representations should be ~0."""
        torch.manual_seed(42)
        X = torch.randn(200, 32)
        Y = torch.randn(200, 32)
        cka = minibatch_cka(X, Y, batch_size=64)
        assert abs(cka) < 0.3  # Should be close to 0

    def test_small_batch_uses_standard(self):
        """Small sample count should use standard CKA."""
        X = torch.randn(50, 16)
        Y = torch.randn(50, 16)
        cka = minibatch_cka(X, Y, batch_size=256)
        # Should not crash, result should be valid
        assert -0.5 <= cka <= 1.5

    def test_cka_bounded(self):
        """CKA should be approximately bounded in [0, 1] for typical inputs."""
        torch.manual_seed(0)
        X = torch.randn(200, 64)
        Y = X @ torch.randn(64, 32)  # Y is a linear transform of X
        cka = minibatch_cka(X, Y, batch_size=64)
        assert -0.1 <= cka <= 1.1  # Allow small numerical slack

    def test_handles_3d_input(self):
        """Should handle 3D inputs by flattening."""
        X = torch.randn(50, 8, 16)  # (batch, seq, dim)
        Y = torch.randn(50, 4, 32)
        cka = minibatch_cka(X, Y, batch_size=256)
        assert isinstance(cka, float)

    def test_consistency_with_different_batch_sizes(self):
        """Results should be approximately consistent across batch sizes."""
        torch.manual_seed(42)
        X = torch.randn(300, 32)
        Y = X + 0.1 * torch.randn(300, 32)

        cka_64 = minibatch_cka(X, Y, batch_size=64)
        cka_128 = minibatch_cka(X, Y, batch_size=128)

        # Should be in the same ballpark
        assert abs(cka_64 - cka_128) < 0.2
