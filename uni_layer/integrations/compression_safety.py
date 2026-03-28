"""
Compression safety audit.

Compares security metrics before and after model compression (pruning,
quantization, distillation) to quantify safety degradation. Identifies
layers where compression increases vulnerability.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CompressionSafetyAudit:
    """
    Audit safety degradation from model compression.

    Compares pre/post-compression security metrics to identify:
    - Which layers became more vulnerable after compression
    - Overall safety degradation score
    - Specific metric regressions

    Example:
        >>> audit = CompressionSafetyAudit(pre_contributions, post_contributions)
        >>> report = audit.audit()
        >>> print(report["overall_degradation"])
    """

    # Security-relevant metric keys
    SECURITY_METRICS = [
        "adv_sensitivity",
        "adv_amplification",
        "adv_directional_change",
        "activation_skewness",
        "activation_kurtosis",
        "neuron_outlier_ratio",
        "mi_risk_score",
        "gradient_entropy",
        "injection_vulnerability",
    ]

    def __init__(
        self,
        pre_contributions: Dict[str, Dict[str, Any]],
        post_contributions: Dict[str, Dict[str, Any]],
    ):
        self.pre = pre_contributions
        self.post = post_contributions
        self.common_layers = sorted(set(self.pre.keys()) & set(self.post.keys()))

    def per_layer_delta(self, metric_name: str) -> Dict[str, float]:
        """
        Compute per-layer change in a metric after compression.

        Positive delta = metric increased (potentially worse for security metrics).
        """
        deltas = {}
        for layer in self.common_layers:
            pre_val = self.pre[layer].get(metric_name)
            post_val = self.post[layer].get(metric_name)
            if pre_val is not None and post_val is not None:
                deltas[layer] = float(post_val - pre_val)
        return deltas

    def degraded_layers(
        self,
        metric_name: str,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find layers where a security metric got worse after compression.

        Args:
            metric_name: Security metric to check
            threshold: Minimum delta to consider as degradation

        Returns:
            List of {layer_name, pre_value, post_value, delta} sorted by delta descending
        """
        results = []
        for layer in self.common_layers:
            pre_val = self.pre[layer].get(metric_name)
            post_val = self.post[layer].get(metric_name)
            if pre_val is not None and post_val is not None:
                delta = post_val - pre_val
                if delta > threshold:
                    results.append(
                        {
                            "layer_name": layer,
                            "pre_value": float(pre_val),
                            "post_value": float(post_val),
                            "delta": float(delta),
                        }
                    )
        results.sort(key=lambda x: x["delta"], reverse=True)
        return results

    def audit(self) -> Dict[str, Any]:
        """
        Run full compression safety audit.

        Returns:
            Dict with overall_degradation, per_metric summaries,
            most_affected_layers, and recommendations
        """
        metric_summaries = {}
        all_degradations = []

        for metric in self.SECURITY_METRICS:
            deltas = self.per_layer_delta(metric)
            if not deltas:
                continue

            values = list(deltas.values())
            mean_delta = np.mean(values)
            max_delta = max(values)
            degraded = sum(1 for v in values if v > 0)

            metric_summaries[metric] = {
                "mean_delta": float(mean_delta),
                "max_delta": float(max_delta),
                "num_degraded_layers": degraded,
                "num_total_layers": len(values),
                "degradation_ratio": degraded / len(values) if values else 0.0,
            }

            if mean_delta > 0:
                all_degradations.append(mean_delta)

        # Overall degradation score (0 = no degradation, 1 = severe)
        if all_degradations:
            overall = min(1.0, np.mean(all_degradations) * 10)  # scale factor
        else:
            overall = 0.0

        # Most affected layers across all metrics
        layer_total_degradation = {}
        for metric in self.SECURITY_METRICS:
            for item in self.degraded_layers(metric):
                name = item["layer_name"]
                layer_total_degradation[name] = layer_total_degradation.get(name, 0) + item["delta"]

        most_affected = sorted(
            [{"layer_name": k, "total_degradation": v} for k, v in layer_total_degradation.items()],
            key=lambda x: x["total_degradation"],
            reverse=True,
        )[:10]

        # Recommendations
        recommendations = []
        if overall > 0.5:
            recommendations.append(
                "HIGH RISK: Compression significantly increased model vulnerability."
            )
        elif overall > 0.2:
            recommendations.append(
                "MODERATE RISK: Some security metrics degraded after compression."
            )
        else:
            recommendations.append("LOW RISK: Compression had minimal impact on security metrics.")

        for ms_name, ms in metric_summaries.items():
            if ms["degradation_ratio"] > 0.5:
                recommendations.append(
                    f"  - {ms_name}: {ms['num_degraded_layers']}/{ms['num_total_layers']} layers degraded"
                )

        return {
            "overall_degradation": float(overall),
            "metric_summaries": metric_summaries,
            "most_affected_layers": most_affected,
            "recommendations": recommendations,
            "num_common_layers": len(self.common_layers),
        }
