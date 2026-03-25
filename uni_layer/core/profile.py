"""
LayerProfile: automatic insight extraction from layer contribution data.

All analyses run on CPU against the already-computed contributions dict.
No additional forward/backward passes. Millisecond-level execution.

Usage:
    contributions = analyzer.compute_metrics(...)
    profile = LayerProfile(contributions)

    profile.redundant_layers        # layers safe to prune
    profile.bottleneck_layers       # representation bottlenecks
    profile.consensus_ranking       # multi-metric agreement ranking
    profile.depth_trends            # how metrics evolve with depth
    profile.anomalies               # statistically unusual layers
    profile.pruning_suggestion(0.3) # actionable pruning plan
    profile.lora_suggestion(8)      # actionable LoRA plan
    profile.summary()               # one-paragraph natural language summary
    profile.to_dict()               # full JSON-serializable report
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import OrderedDict


class LayerProfile:
    """
    Automatic insight extraction from layer contribution analysis.

    Takes the output of LayerAnalyzer.compute_metrics() and produces
    actionable analyses without any additional GPU computation.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()
        model_name: Optional model name for reports
    """

    def __init__(
        self,
        contributions: Dict[str, Dict[str, Any]],
        model_name: str = "unknown",
    ):
        self.contributions = contributions
        self.model_name = model_name
        self._layers = list(contributions.keys())
        self._n_layers = len(self._layers)

        # Extract available metric keys (exclude metadata)
        self._metric_keys = self._detect_metric_keys()
        self._matrix = self._build_matrix()  # (n_layers, n_metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_metric_keys(self) -> List[str]:
        """Find all float-valued metric keys present in contributions."""
        meta = {"layer_idx", "layer_type"}
        keys = set()
        for metrics in self.contributions.values():
            for k, v in metrics.items():
                if k not in meta and isinstance(v, (int, float)) and v is not None:
                    keys.add(k)
        # Stable order: sort
        return sorted(keys)

    def _build_matrix(self) -> np.ndarray:
        """Build (n_layers, n_metrics) matrix from contributions."""
        mat = np.zeros((self._n_layers, len(self._metric_keys)))
        for i, layer in enumerate(self._layers):
            for j, key in enumerate(self._metric_keys):
                val = self.contributions[layer].get(key)
                mat[i, j] = float(val) if val is not None else 0.0
        return mat

    def _get_col(self, metric_key: str) -> Optional[np.ndarray]:
        """Get column vector for a metric. Returns None if not available."""
        if metric_key not in self._metric_keys:
            return None
        j = self._metric_keys.index(metric_key)
        return self._matrix[:, j]

    def _normalize_col(self, col: np.ndarray) -> np.ndarray:
        """Min-max normalize a column to [0, 1]."""
        mn, mx = col.min(), col.max()
        if mx - mn < 1e-12:
            return np.zeros_like(col)
        return (col - mn) / (mx - mn)

    # ------------------------------------------------------------------
    # Analysis properties (lazy-evaluated, cached)
    # ------------------------------------------------------------------

    @property
    def redundant_layers(self) -> List[str]:
        """
        Layers with low transformation (high similarity between input/output).
        Detected via BlockInfluence < 0.02 or bottom 10% of gradient_norm.
        """
        result = []

        bi = self._get_col("block_influence")
        if bi is not None:
            threshold = max(0.02, np.percentile(bi[bi > 0], 10) if np.any(bi > 0) else 0.02)
            for i, val in enumerate(bi):
                if 0 < val < threshold:
                    result.append(self._layers[i])
            if result:
                return result

        # Fallback: bottom 10% by gradient_norm
        gn = self._get_col("gradient_norm")
        if gn is not None and np.any(gn > 0):
            threshold = np.percentile(gn[gn > 0], 10)
            for i, val in enumerate(gn):
                if 0 < val < threshold:
                    result.append(self._layers[i])

        return result

    @property
    def bottleneck_layers(self) -> List[str]:
        """
        Layers where effective_rank drops significantly relative to neighbors.
        A bottleneck constrains information flow.
        """
        er = self._get_col("effective_rank")
        if er is None or self._n_layers < 3:
            return []

        result = []
        for i in range(1, self._n_layers - 1):
            if er[i] <= 0:
                continue
            avg_neighbor = (er[i - 1] + er[i + 1]) / 2
            if avg_neighbor > 0 and er[i] / avg_neighbor < 0.7:
                result.append(self._layers[i])

        return result

    @property
    def consensus_ranking(self) -> List[Dict[str, Any]]:
        """
        Multi-metric ranking using Borda count.
        Each metric votes by ranking layers; scores are averaged across metrics.
        """
        if self._n_layers == 0 or len(self._metric_keys) == 0:
            return []

        # Borda: rank each metric column, higher value = higher rank
        borda = np.zeros(self._n_layers)
        n_voted = 0

        for j, key in enumerate(self._metric_keys):
            col = self._matrix[:, j]
            if np.all(col == 0) or np.all(np.isnan(col)):
                continue
            # Rank: argsort of argsort gives rank (0 = lowest)
            ranks = np.argsort(np.argsort(col)).astype(float)
            borda += ranks
            n_voted += 1

        if n_voted == 0:
            return []

        borda /= n_voted
        # Normalize to [0, 1]
        borda = borda / max(self._n_layers - 1, 1)

        order = np.argsort(-borda)
        result = []
        for rank, idx in enumerate(order):
            entry = {
                "layer": self._layers[idx],
                "rank": rank + 1,
                "consensus_score": round(float(borda[idx]), 4),
            }
            # Add per-metric normalized scores
            scores = {}
            for j, key in enumerate(self._metric_keys):
                val = self._matrix[idx, j]
                if val != 0:
                    scores[key] = round(float(val), 6)
            entry["scores"] = scores
            result.append(entry)

        return result

    @property
    def depth_trends(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how each metric changes with layer depth.
        Reports trend type (increasing/decreasing/U-shaped/flat) and early/mid/late averages.
        """
        if self._n_layers < 3:
            return {}

        third = max(self._n_layers // 3, 1)
        result = {}

        for j, key in enumerate(self._metric_keys):
            col = self._matrix[:, j]
            if np.all(col == 0):
                continue

            early = float(np.mean(col[:third]))
            mid = float(np.mean(col[third:2 * third]))
            late = float(np.mean(col[2 * third:]))

            # Determine trend
            if early > mid and late > mid and (early > mid * 1.2 or late > mid * 1.2):
                trend = "U-shaped"
            elif early < mid and late < mid and (early < mid * 0.8 or late < mid * 0.8):
                trend = "inverted-U"
            elif late > early * 1.15:
                trend = "increasing"
            elif early > late * 1.15:
                trend = "decreasing"
            else:
                trend = "flat"

            result[key] = {
                "trend": trend,
                "early": round(early, 6),
                "middle": round(mid, 6),
                "late": round(late, 6),
            }

        return result

    @property
    def anomalies(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Layers with z-score > 2 on any metric (statistically unusual).
        """
        result = {}

        for j, key in enumerate(self._metric_keys):
            col = self._matrix[:, j]
            nonzero = col[col != 0]
            if len(nonzero) < 3:
                continue

            mean = np.mean(nonzero)
            std = np.std(nonzero)
            if std < 1e-12:
                continue

            for i in range(self._n_layers):
                if col[i] == 0:
                    continue
                z = (col[i] - mean) / std
                if abs(z) > 2.0:
                    layer = self._layers[i]
                    if layer not in result:
                        result[layer] = []
                    result[layer].append({
                        "metric": key,
                        "value": round(float(col[i]), 6),
                        "z_score": round(float(z), 2),
                        "note": "unusually high" if z > 0 else "unusually low",
                    })

        return result

    @property
    def layer_clusters(self) -> Dict[str, List[str]]:
        """
        Cluster layers into groups by their metric profiles.
        Uses simple k-means-style grouping (no sklearn dependency).
        """
        if self._n_layers < 3 or len(self._metric_keys) == 0:
            return {}

        # Normalize each column
        normed = np.zeros_like(self._matrix)
        for j in range(self._matrix.shape[1]):
            normed[:, j] = self._normalize_col(self._matrix[:, j])

        # Simple 3-group split by consensus score
        borda = np.zeros(self._n_layers)
        for j in range(normed.shape[1]):
            borda += np.argsort(np.argsort(normed[:, j])).astype(float)
        borda /= max(normed.shape[1], 1)

        # Tercile split
        t1 = np.percentile(borda, 33)
        t2 = np.percentile(borda, 67)

        groups = {"high_contribution": [], "medium_contribution": [], "low_contribution": []}
        for i, score in enumerate(borda):
            if score >= t2:
                groups["high_contribution"].append(self._layers[i])
            elif score >= t1:
                groups["medium_contribution"].append(self._layers[i])
            else:
                groups["low_contribution"].append(self._layers[i])

        return {k: v for k, v in groups.items() if v}

    # ------------------------------------------------------------------
    # Actionable suggestions
    # ------------------------------------------------------------------

    def pruning_suggestion(self, target_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Generate a pruning plan based on layer analysis.

        Args:
            target_ratio: Fraction of layers to prune (0-1)

        Returns:
            Dict with safe_to_remove layers and estimated speedup
        """
        n_remove = max(1, int(self._n_layers * target_ratio))

        # Use consensus ranking (lowest = least important)
        ranking = self.consensus_ranking
        if not ranking:
            return {"safe_to_remove": [], "estimated_speedup": "0%"}

        # Take bottom-N layers
        candidates = ranking[-n_remove:]
        safe = [c["layer"] for c in reversed(candidates)]

        speedup = len(safe) / max(self._n_layers, 1) * 100
        return {
            "safe_to_remove": safe,
            "num_layers_removed": len(safe),
            "estimated_speedup": f"{speedup:.1f}%",
        }

    def lora_suggestion(self, base_rank: int = 8, max_rank: int = 64) -> Dict[str, Any]:
        """
        Suggest LoRA targets and adaptive ranks.

        Args:
            base_rank: Minimum LoRA rank
            max_rank: Maximum LoRA rank

        Returns:
            Dict with target layers and per-layer ranks
        """
        ranking = self.consensus_ranking
        if not ranking:
            return {"target_layers": [], "adaptive_ranks": {}}

        # Top 50% of layers get LoRA
        n_target = max(1, self._n_layers // 2)
        targets = ranking[:n_target]

        # Adaptive rank: higher contribution = higher rank
        ranks = {}
        for entry in targets:
            score = entry["consensus_score"]
            rank = int(base_rank + (max_rank - base_rank) * score)
            rank = max(base_rank, (rank // 8) * 8)  # Round to multiple of 8
            ranks[entry["layer"]] = rank

        return {
            "target_layers": [t["layer"] for t in targets],
            "adaptive_ranks": ranks,
            "total_lora_layers": len(ranks),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a one-paragraph natural language summary."""
        parts = []
        parts.append(f"{self._n_layers}-layer model")

        if self.model_name != "unknown":
            parts[0] = f"{self.model_name} ({parts[0]})"

        parts.append(f"analyzed with {len(self._metric_keys)} metrics")

        # Depth trend
        trends = self.depth_trends
        gn_trend = trends.get("gradient_norm", {}).get("trend")
        if gn_trend:
            parts.append(f"{gn_trend} gradient norm distribution")

        # Redundancy
        redundant = self.redundant_layers
        if redundant:
            names = ", ".join(redundant[:3])
            parts.append(
                f"{len(redundant)} redundant layer{'s' if len(redundant) > 1 else ''} "
                f"({names}{'...' if len(redundant) > 3 else ''}) safe to prune"
            )

        # Bottleneck
        bottlenecks = self.bottleneck_layers
        if bottlenecks:
            parts.append(
                f"{len(bottlenecks)} bottleneck layer{'s' if len(bottlenecks) > 1 else ''} "
                f"({', '.join(bottlenecks[:2])})"
            )

        # Top layers
        ranking = self.consensus_ranking
        if ranking:
            top = ranking[0]["layer"]
            bot = ranking[-1]["layer"]
            parts.append(f"most important: {top}, least important: {bot}")

        # Anomalies
        anomalies = self.anomalies
        if anomalies:
            parts.append(f"{len(anomalies)} layer{'s' if len(anomalies) > 1 else ''} with anomalous metrics")

        return ". ".join(parts) + "."

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export full profile as JSON-serializable dict."""
        return {
            "model": self.model_name,
            "num_layers": self._n_layers,
            "metrics_used": self._metric_keys,
            "analysis": {
                "redundant_layers": self.redundant_layers,
                "bottleneck_layers": self.bottleneck_layers,
                "consensus_ranking": self.consensus_ranking,
                "depth_trends": self.depth_trends,
                "anomalies": self.anomalies,
                "layer_clusters": self.layer_clusters,
                "suggestions": {
                    "pruning": self.pruning_suggestion(),
                    "lora": self.lora_suggestion(),
                },
            },
            "summary": self.summary(),
        }

    def __repr__(self) -> str:
        return (
            f"LayerProfile(model='{self.model_name}', layers={self._n_layers}, "
            f"metrics={len(self._metric_keys)})"
        )
