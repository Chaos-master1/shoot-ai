"""Tests for Injury Risk Prediction module."""

from kawkab.core.injury_risk import InjuryRiskPredictor


class TestComputeAcwrOverload:
    def test_normal_acwr(self):
        irp = InjuryRiskPredictor()
        workload = [100] * 28
        result = irp.compute_acwr_overload(workload)
        assert result["acwr"] == 1.0
        assert result["risk_level"] == "moderate"

    def test_high_acwr(self):
        irp = InjuryRiskPredictor()
        workload = [50] * 21 + [150] * 7
        result = irp.compute_acwr_overload(workload)
        assert result["acwr"] > 1.3

    def test_insufficient_data(self):
        irp = InjuryRiskPredictor()
        result = irp.compute_acwr_overload([100, 100, 100])
        assert result["risk_level"] == "low"
        assert "Insufficient" in result["recommendation"]

    def test_empty_data(self):
        irp = InjuryRiskPredictor()
        result = irp.compute_acwr_overload([])
        assert result["acwr"] == 0.0


class TestPredictInjuryRisk:
    def test_low_risk_profile(self):
        irp = InjuryRiskPredictor()
        profile = {
            "acwr": 1.0,
            "recent_sprint_count": 5,
            "recent_distance_km": 3.0,
            "fatigue_index": 2.0,
            "position": "GK",
            "days_since_last_rest": 1,
        }
        result = irp.predict_injury_risk(profile)
        assert result["risk_score"] < 0.4
        assert result["risk_category"] in ("low", "moderate")

    def test_high_risk_profile(self):
        irp = InjuryRiskPredictor()
        profile = {
            "acwr": 1.6,
            "recent_sprint_count": 30,
            "recent_distance_km": 12.0,
            "fatigue_index": 25.0,
            "position": "MID",
            "days_since_last_rest": 15,
        }
        result = irp.predict_injury_risk(profile)
        assert result["risk_score"] >= 0.4
        assert result["risk_category"] in ("high", "critical")

    def test_risk_factors_populated(self):
        irp = InjuryRiskPredictor()
        profile = {
            "acwr": 1.6,
            "recent_sprint_count": 28,
            "recent_distance_km": 10.0,
            "fatigue_index": 20.0,
            "position": "FWD",
            "days_since_last_rest": 12,
        }
        result = irp.predict_injury_risk(profile)
        assert len(result["key_risk_factors"]) > 0

    def test_minimal_profile(self):
        irp = InjuryRiskPredictor()
        profile = {
            "acwr": 1.0,
            "recent_sprint_count": 0,
            "recent_distance_km": 0,
            "fatigue_index": 0,
            "position": "DEF",
            "days_since_last_rest": 1,
        }
        result = irp.predict_injury_risk(profile)
        assert 0 <= result["risk_score"] <= 1


class TestComputeRecoveryRecommendation:
    def test_full_training_low_risk(self):
        irp = InjuryRiskPredictor()
        rec = irp.compute_recovery_recommendation(0.1, "MID")
        assert rec == "full training"

    def test_rest_day_critical(self):
        irp = InjuryRiskPredictor()
        rec = irp.compute_recovery_recommendation(0.8, "MID")
        assert "rest" in rec or "assessment" in rec

    def test_modified_training_high_risk(self):
        irp = InjuryRiskPredictor()
        rec = irp.compute_recovery_recommendation(0.5, "DEF")
        assert rec != "full training"  # high risk

    def test_position_specific(self):
        irp = InjuryRiskPredictor()
        rec = irp.compute_recovery_recommendation(0.4, "GK")
        assert isinstance(rec, str)
        assert len(rec) > 0
