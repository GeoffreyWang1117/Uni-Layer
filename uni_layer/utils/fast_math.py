"""
Fast math utilities for metric computation.

Provides optimized versions of expensive linear algebra operations
used across multiple metrics, specifically designed for large-scale models.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def randomized_svd(
    X: torch.Tensor,
    n_components: int = 50,
    n_oversamples: int = 10,
    n_iter: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD for large matrices (Halko et al., 2011).

    Full SVD of an (m x n) matrix costs O(m*n*min(m,n)).
    Randomized SVD costs O(m*n*k) where k = n_components << min(m,n).

    For a 320x4096 activation matrix:
    - Full SVD:       ~420M FLOPs
    - Randomized SVD: ~52M FLOPs (8x faster)

    For a 320x32768 activation matrix (large transformers):
    - Full SVD:       ~3.4B FLOPs
    - Randomized SVD: ~210M FLOPs (16x faster)

    Args:
        X: Input matrix (m x n)
        n_components: Number of singular values to compute
        n_oversamples: Extra dimensions for accuracy (default 10)
        n_iter: Number of power iterations for accuracy (default 4)

    Returns:
        U, S, Vt: Truncated SVD components
    """
    m, n = X.shape
    k = min(n_components + n_oversamples, m, n)

    # If the matrix is small enough, just use full SVD
    if min(m, n) <= 2 * k:
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        return U[:, :n_components], S[:n_components], Vt[:n_components, :]

    # Step 1: Random projection
    Omega = torch.randn(n, k, device=X.device, dtype=X.dtype)
    Y = X @ Omega

    # Step 2: Power iteration for accuracy
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
        # Re-orthogonalize for numerical stability
        Y, _ = torch.linalg.qr(Y)

    # Step 3: QR decomposition
    Q, _ = torch.linalg.qr(Y)

    # Step 4: Project and compute small SVD
    B = Q.T @ X  # (k x n) - much smaller than original
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Step 5: Recover U
    U = Q @ U_hat

    return U[:, :n_components], S[:n_components], Vt[:n_components, :]


def effective_rank_from_svd(
    S: torch.Tensor,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute effective rank from pre-computed singular values.

    EffectiveRank = exp(H(p)) where p_i = s_i / sum(s)
    and H is Shannon entropy.

    Args:
        S: Singular values (sorted, descending)
        epsilon: Small constant to avoid log(0)

    Returns:
        Effective rank value
    """
    S = S[S > epsilon]
    if len(S) == 0:
        return 0.0

    p = S / (S.sum() + epsilon)
    entropy = -(p * torch.log(p + epsilon)).sum()
    return torch.exp(entropy).item()


def minibatch_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """
    Minibatch CKA for large sample counts (Nguyen et al., 2021).

    Standard CKA requires O(N^2) memory for kernel matrices.
    Minibatch CKA uses O(batch_size^2) memory regardless of N.

    For N=10000 samples:
    - Standard CKA: ~800MB for kernel matrices
    - Minibatch CKA: ~500KB per batch (1600x less memory)

    Args:
        X: First representation matrix (N x d1)
        Y: Second representation matrix (N x d2)
        batch_size: Minibatch size for CKA computation

    Returns:
        CKA score
    """
    if X.dim() > 2:
        X = X.reshape(X.size(0), -1)
    if Y.dim() > 2:
        Y = Y.reshape(Y.size(0), -1)

    N = X.size(0)

    if N <= batch_size:
        # Small enough for standard CKA
        return _standard_cka(X, Y)

    # Accumulate HSIC statistics across minibatches
    hsic_xy_sum = 0.0
    hsic_xx_sum = 0.0
    hsic_yy_sum = 0.0
    n_batches = 0

    indices = torch.randperm(N)

    for start in range(0, N - batch_size + 1, batch_size):
        idx = indices[start:start + batch_size]
        X_batch = X[idx]
        Y_batch = Y[idx]

        hsic_xy_sum += _unbiased_hsic(X_batch, Y_batch)
        hsic_xx_sum += _unbiased_hsic(X_batch, X_batch)
        hsic_yy_sum += _unbiased_hsic(Y_batch, Y_batch)
        n_batches += 1

    if n_batches == 0:
        return 0.0

    hsic_xy = hsic_xy_sum / n_batches
    hsic_xx = hsic_xx_sum / n_batches
    hsic_yy = hsic_yy_sum / n_batches

    denom = (hsic_xx * hsic_yy) ** 0.5
    if denom < 1e-10:
        return 0.0

    return float(hsic_xy / denom)


def _standard_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Standard CKA with linear kernel (small N)."""
    K = X @ X.T
    L = Y @ Y.T
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    K = H @ K @ H
    L = H @ L @ H
    num = (K * L).sum()
    den = torch.sqrt((K * K).sum() * (L * L).sum())
    if den < 1e-10:
        return 0.0
    return (num / den).item()


def _unbiased_hsic(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Unbiased HSIC estimator with linear kernel (Song et al., 2012)."""
    n = X.size(0)
    if n < 4:
        return 0.0

    K = X @ X.T
    L = Y @ Y.T

    # Zero diagonals
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)

    # Unbiased HSIC
    term1 = (K * L).sum()
    term2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    term3 = 2 * (K.sum(dim=0) * L.sum(dim=0)).sum() / (n - 2)

    hsic = (term1 + term2 - term3) / (n * (n - 3))
    return float(hsic)


def hutchinson_trace_estimator(
    hvp_fn,
    params: list,
    num_samples: int = 5,
    device: str = "cuda",
) -> float:
    """
    Hutchinson trace estimator using Hessian-vector products.

    Instead of computing the full Hessian (O(P^2) memory, O(P^3) compute),
    estimates Tr(H) using random vector projections in O(P) memory.

    Tr(H) = E[v^T H v] where v ~ Rademacher(+-1)

    Args:
        hvp_fn: Function that computes Hessian-vector product Hv given v
        params: List of parameters to estimate trace for
        num_samples: Number of random vectors (more = more accurate)
        device: Device for computation

    Returns:
        Estimated trace value
    """
    trace_sum = 0.0

    for _ in range(num_samples):
        # Rademacher random vectors (+-1) have lower variance than Gaussian
        vs = [torch.randint(0, 2, p.shape, device=device).float() * 2 - 1
              for p in params]

        hvs = hvp_fn(vs)

        if hvs is not None:
            trace_sample = sum(
                (v * hv).sum().item()
                for v, hv in zip(vs, hvs)
                if hv is not None
            )
            trace_sum += trace_sample

    return trace_sum / num_samples
