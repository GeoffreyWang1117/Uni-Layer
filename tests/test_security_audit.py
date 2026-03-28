"""Tests for compression safety audit and security_report (v0.6.0 Features 5-6)."""

import pytest


@pytest.fixture
def pre_contributions():
    return {
        "layer.0": {
            "layer_idx": 0,
            "layer_type": "linear",
            "adv_sensitivity": 0.1,
            "neuron_outlier_ratio": 0.02,
            "mi_risk_score": 0.3,
            "injection_vulnerability": 0.2,
        },
        "layer.1": {
            "layer_idx": 1,
            "layer_type": "linear",
            "adv_sensitivity": 0.05,
            "neuron_outlier_ratio": 0.01,
            "mi_risk_score": 0.2,
            "injection_vulnerability": 0.1,
        },
        "layer.2": {
            "layer_idx": 2,
            "layer_type": "linear",
            "adv_sensitivity": 0.2,
            "neuron_outlier_ratio": 0.03,
            "mi_risk_score": 0.5,
            "injection_vulnerability": 0.4,
        },
    }


@pytest.fixture
def post_contributions():
    """Post-compression: some metrics got worse."""
    return {
        "layer.0": {
            "layer_idx": 0,
            "layer_type": "linear",
            "adv_sensitivity": 0.3,
            "neuron_outlier_ratio": 0.05,
            "mi_risk_score": 0.4,
            "injection_vulnerability": 0.3,
        },
        "layer.1": {
            "layer_idx": 1,
            "layer_type": "linear",
            "adv_sensitivity": 0.04,
            "neuron_outlier_ratio": 0.01,
            "mi_risk_score": 0.15,
            "injection_vulnerability": 0.08,
        },
        "layer.2": {
            "layer_idx": 2,
            "layer_type": "linear",
            "adv_sensitivity": 0.5,
            "neuron_outlier_ratio": 0.1,
            "mi_risk_score": 0.7,
            "injection_vulnerability": 0.6,
        },
    }


class TestCompressionSafetyAudit:
    def test_import(self):
        from uni_layer.integrations import CompressionSafetyAudit

        assert CompressionSafetyAudit is not None

    def test_per_layer_delta(self, pre_contributions, post_contributions):
        from uni_layer.integrations import CompressionSafetyAudit

        audit = CompressionSafetyAudit(pre_contributions, post_contributions)
        deltas = audit.per_layer_delta("adv_sensitivity")

        assert "layer.0" in deltas
        assert abs(deltas["layer.0"] - 0.2) < 1e-6  # 0.3 - 0.1
        assert deltas["layer.1"] < 0  # improved

    def test_degraded_layers(self, pre_contributions, post_contributions):
        from uni_layer.integrations import CompressionSafetyAudit

        audit = CompressionSafetyAudit(pre_contributions, post_contributions)
        degraded = audit.degraded_layers("adv_sensitivity")

        assert len(degraded) >= 1
        assert degraded[0]["layer_name"] in ("layer.0", "layer.2")
        assert degraded[0]["delta"] > 0

    def test_audit(self, pre_contributions, post_contributions):
        from uni_layer.integrations import CompressionSafetyAudit

        audit = CompressionSafetyAudit(pre_contributions, post_contributions)
        report = audit.audit()

        assert "overall_degradation" in report
        assert "metric_summaries" in report
        assert "most_affected_layers" in report
        assert "recommendations" in report
        assert report["overall_degradation"] >= 0
        assert report["num_common_layers"] == 3

    def test_no_degradation(self, pre_contributions):
        """If pre == post, no degradation."""
        from uni_layer.integrations import CompressionSafetyAudit

        audit = CompressionSafetyAudit(pre_contributions, pre_contributions)
        report = audit.audit()
        assert report["overall_degradation"] == 0.0

    def test_recommendations(self, pre_contributions, post_contributions):
        from uni_layer.integrations import CompressionSafetyAudit

        audit = CompressionSafetyAudit(pre_contributions, post_contributions)
        report = audit.audit()
        assert len(report["recommendations"]) >= 1


class TestSecurityReport:
    def test_no_security_metrics(self):
        from uni_layer.core.profile import LayerProfile

        contrib = {
            "layer.0": {"layer_idx": 0, "layer_type": "linear", "gradient_norm": 0.5},
        }
        profile = LayerProfile(contrib)
        report = profile.security_report()

        assert report["categories_found"] == []
        assert "No security metrics" in report["summary"]

    def test_with_security_metrics(self, pre_contributions):
        from uni_layer.core.profile import LayerProfile

        profile = LayerProfile(pre_contributions, model_name="test-model")
        report = profile.security_report()

        assert len(report["categories_found"]) > 0
        assert "adversarial" in report["categories_found"]
        assert len(report["layer_risks"]) == 3
        assert len(report["top_risks"]) <= 5

    def test_composite_risk(self, pre_contributions):
        from uni_layer.core.profile import LayerProfile

        profile = LayerProfile(pre_contributions)
        report = profile.security_report()

        for layer_name, risk in report["layer_risks"].items():
            assert "composite_risk" in risk
            assert "max_risk" in risk
            assert 0.0 <= risk["composite_risk"] <= 1.0

    def test_top_risks_sorted(self, pre_contributions):
        from uni_layer.core.profile import LayerProfile

        profile = LayerProfile(pre_contributions)
        report = profile.security_report()

        if len(report["top_risks"]) >= 2:
            risks = [r["composite_risk"] for r in report["top_risks"]]
            assert risks == sorted(risks, reverse=True)

    def test_summary_text(self, pre_contributions):
        from uni_layer.core.profile import LayerProfile

        profile = LayerProfile(pre_contributions)
        report = profile.security_report()
        assert "layer" in report["summary"].lower()
        assert "risk" in report["summary"].lower()
