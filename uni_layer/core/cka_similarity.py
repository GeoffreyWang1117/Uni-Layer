"""
Layer-to-layer CKA similarity matrix.

Computes pairwise Centered Kernel Alignment (CKA) between all layers,
producing an N×N similarity matrix. This reveals which layers learn
similar representations (redundancy) and which are distinct (diversity).

Reference:
    Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019
    Nguyen et al., "Do Wide Neural Networks Learn the Same Things?", ICLR 2021
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from uni_layer.core.cache import ActivationCache, BatchCache
from uni_layer.utils.fast_math import _standard_cka, minibatch_cka
from uni_layer.utils.layer_utils import get_model_layers
from uni_layer.utils.model_adapter import model_forward


class CKASimilarity:
    """
    Compute pairwise CKA similarity between all layers in a model.

    Produces an N×N matrix where entry (i, j) is the CKA score between
    layer i and layer j. The diagonal is always 1.0 (self-similarity).

    Uses a single forward pass to capture all layer activations, then
    computes CKA pairwise. For large sample counts, automatically uses
    minibatch CKA to bound memory usage.

    Example:
        >>> cka_sim = CKASimilarity(model)
        >>> matrix, layer_names = cka_sim.compute(data_loader)
        >>> print(matrix.shape)  # (num_layers, num_layers)
        >>> # Find most redundant layer pair
        >>> i, j, score = cka_sim.most_similar_pair(matrix, layer_names)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[Dict[str, nn.Module]] = None,
        kernel: str = "linear",
        cka_batch_size: int = 256,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: PyTorch model to analyze
            layers: Dict of {name: module} to analyze. If None, auto-extracts.
            kernel: Kernel type for CKA ('linear' or 'rbf')
            cka_batch_size: Batch size for minibatch CKA (controls memory)
            device: Device for computation (default: auto-detect)
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.kernel = kernel
        self.cka_batch_size = cka_batch_size

        if layers is not None:
            self.layers = layers
        else:
            self.layers = get_model_layers(model)

        self.layer_names = list(self.layers.keys())

    def compute(
        self,
        data_loader: Any,
        num_batches: int = 10,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute the N×N CKA similarity matrix.

        Args:
            data_loader: DataLoader providing input data
            num_batches: Number of batches to use for activation capture
            verbose: Whether to print progress

        Returns:
            Tuple of (similarity_matrix, layer_names) where
            similarity_matrix is shape (N, N) with values in [0, 1]
        """
        n = len(self.layer_names)
        if verbose:
            print(f"Computing {n}×{n} CKA similarity matrix...")

        # Phase 1: Capture all layer activations in one forward pass
        batch_cache = BatchCache(data_loader, num_batches=num_batches, device=self.device)
        act_cache = ActivationCache(self.model, self.layers)
        act_cache.capture(batch_cache, num_batches=num_batches, device=self.device)

        # Phase 2: Flatten activations per layer
        flat_activations = OrderedDict()
        for name in self.layer_names:
            act = act_cache.get_concatenated(name)
            if act is not None:
                if act.dim() > 2:
                    act = act.reshape(act.size(0), -1)
                flat_activations[name] = act
            else:
                flat_activations[name] = None

        del act_cache

        # Phase 3: Compute pairwise CKA
        matrix = np.eye(n, dtype=np.float64)

        total_pairs = n * (n - 1) // 2
        computed = 0

        for i in range(n):
            for j in range(i + 1, n):
                X = flat_activations[self.layer_names[i]]
                Y = flat_activations[self.layer_names[j]]

                if X is None or Y is None:
                    score = 0.0
                else:
                    N_samples = X.size(0)
                    if N_samples > self.cka_batch_size and self.kernel == "linear":
                        score = minibatch_cka(X, Y, batch_size=self.cka_batch_size)
                    else:
                        score = _standard_cka(X, Y)

                matrix[i, j] = score
                matrix[j, i] = score
                computed += 1

                if verbose and computed % max(1, total_pairs // 5) == 0:
                    print(f"  Progress: {computed}/{total_pairs} pairs")

        if verbose:
            print(f"  Done. Matrix shape: {matrix.shape}")

        return matrix, self.layer_names

    def most_similar_pairs(
        self,
        matrix: np.ndarray,
        layer_names: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Find the most similar layer pairs (potential redundancy).

        Args:
            matrix: CKA similarity matrix from compute()
            layer_names: Layer names from compute()
            top_k: Number of top pairs to return

        Returns:
            List of (layer_i, layer_j, cka_score) sorted by similarity descending
        """
        n = len(layer_names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((layer_names[i], layer_names[j], matrix[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]

    def most_distinct_pairs(
        self,
        matrix: np.ndarray,
        layer_names: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Find the most distinct layer pairs (highest representation diversity).

        Args:
            matrix: CKA similarity matrix from compute()
            layer_names: Layer names from compute()
            top_k: Number of top pairs to return

        Returns:
            List of (layer_i, layer_j, cka_score) sorted by similarity ascending
        """
        n = len(layer_names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((layer_names[i], layer_names[j], matrix[i, j]))

        pairs.sort(key=lambda x: x[2])
        return pairs[:top_k]

    def redundant_layers(
        self,
        matrix: np.ndarray,
        layer_names: List[str],
        threshold: float = 0.95,
    ) -> List[Tuple[str, str, float]]:
        """
        Find layer pairs with CKA above threshold (nearly identical representations).

        These layers are strong candidates for pruning — removing one from a
        highly similar pair should have minimal impact on model behavior.

        Args:
            matrix: CKA similarity matrix from compute()
            layer_names: Layer names from compute()
            threshold: CKA threshold for considering layers redundant

        Returns:
            List of (layer_i, layer_j, cka_score) for redundant pairs
        """
        n = len(layer_names)
        redundant = []
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] >= threshold:
                    redundant.append((layer_names[i], layer_names[j], matrix[i, j]))

        redundant.sort(key=lambda x: x[2], reverse=True)
        return redundant

    def layer_uniqueness(
        self,
        matrix: np.ndarray,
        layer_names: List[str],
    ) -> Dict[str, float]:
        """
        Compute a uniqueness score for each layer.

        Uniqueness = 1 - mean(CKA with all other layers).
        Higher uniqueness means the layer learns distinct representations.

        Args:
            matrix: CKA similarity matrix from compute()
            layer_names: Layer names from compute()

        Returns:
            Dict mapping layer name to uniqueness score in [0, 1]
        """
        n = len(layer_names)
        uniqueness = {}
        for i in range(n):
            # Mean CKA with all other layers (exclude self-similarity)
            other_scores = [matrix[i, j] for j in range(n) if j != i]
            mean_sim = np.mean(other_scores) if other_scores else 0.0
            uniqueness[layer_names[i]] = 1.0 - mean_sim

        return uniqueness

    def to_dict(
        self,
        matrix: np.ndarray,
        layer_names: List[str],
    ) -> Dict[str, Any]:
        """
        Export full results as a JSON-serializable dictionary.

        Returns:
            Dict with matrix, layer_names, redundant_pairs, and uniqueness scores
        """
        return {
            "matrix": matrix.tolist(),
            "layer_names": layer_names,
            "most_similar": [
                {"layer_i": a, "layer_j": b, "cka": c}
                for a, b, c in self.most_similar_pairs(matrix, layer_names)
            ],
            "most_distinct": [
                {"layer_i": a, "layer_j": b, "cka": c}
                for a, b, c in self.most_distinct_pairs(matrix, layer_names)
            ],
            "redundant_pairs": [
                {"layer_i": a, "layer_j": b, "cka": c}
                for a, b, c in self.redundant_layers(matrix, layer_names)
            ],
            "layer_uniqueness": self.layer_uniqueness(matrix, layer_names),
        }
