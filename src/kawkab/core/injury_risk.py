"""Injury Risk Prediction.

Computes ACWR, injury risk scores using a weighted heuristic
formula, and provides recovery recommendations. All numpy-only.
"""

from __future__ import annotations

from typing import Any

ACWR_RISK_THRESHOLDS = [
    ("low", 0.0, 0.8),
    ("moderate", 0.8, 1.3),
    ("high", 1.3, 1.5),
    ("critical", 1.5, float("inf")),
]

RECOVERY_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "GK": {
        "low": "full training",
        "moderate": "full training",
        "high": "modified training",
        "critical": "rest day",
    },
    "DEF": {
        "low": "full training",
        "moderate": "full training",
        "high": "modified training",
        "critical": "rest day",
    },
    "MID": {
        "low": "full training",
        "moderate": "full training",
        "high": "modified training",
        "critical": "rest day",
    },
    "FWD": {
        "low": "full training",
        "moderate": "full training",
        "high": "modified training",
        "critical": "rest day",
    },
}

DEFAULT_RECOMMENDATION = {
    "low": "full training",
    "moderate": "full training",
    "high": "modified training",
    "critical": "medical assessment",
}


class InjuryRiskPredictor:
    def compute_acwr_overload(self, workload_data: list[float]) -> dict[str, Any]:
        if len(workload_data) < 7:
            return {
                "acwr": 0.0,
                "risk_level": "low",
                "recommendation": "Insufficient data (need 7+ days)",
            }
        acute_window = 7
        chronic_window = 28
        acute = (
            sum(workload_data[-acute_window:]) / acute_window
            if len(workload_data) >= acute_window
            else sum(workload_data) / len(workload_data)
        )
        chronic_data = (
            workload_data[-chronic_window:]
            if len(workload_data) >= chronic_window
            else workload_data
        )
        chronic = sum(chronic_data) / len(chronic_data) if chronic_data else 1.0
        acwr = acute / chronic if chronic > 0 else 1.0
        risk_level = "low"
        for level, lo, hi in ACWR_RISK_THRESHOLDS:
            if lo <= acwr < hi:
                risk_level = level
                break
        rec = DEFAULT_RECOMMENDATION.get(risk_level, "full training")
        return {"acwr": round(acwr, 3), "risk_level": risk_level, "recommendation": rec}

    def predict_injury_risk(self, player_profile: dict[str, Any]) -> dict[str, Any]:
        acwr = float(player_profile.get("acwr", 1.0))
        recent_sprints = int(player_profile.get("recent_sprint_count", 0))
        recent_distance = float(player_profile.get("recent_distance_km", 0))
        fatigue_index = float(player_profile.get("fatigue_index", 0))
        position = str(player_profile.get("position", "MID"))
        days_since_rest = int(player_profile.get("days_since_last_rest", 0))
        score = 0.0
        if acwr > 1.5:
            score += 0.3
        elif acwr > 1.3:
            score += 0.2
        elif acwr < 0.5:
            score += 0.15
        sprint_norm = min(recent_sprints / 30.0, 1.0)
        score += sprint_norm * 0.2
        dist_norm = min(recent_distance / 12.0, 1.0)
        score += dist_norm * 0.15
        score += min(fatigue_index / 30.0, 1.0) * 0.15
        position_risk = {"GK": 0.05, "DEF": 0.1, "MID": 0.15, "FWD": 0.12}
        score += position_risk.get(position, 0.1)
        if days_since_rest > 14:
            score += 0.1
        elif days_since_rest > 7:
            score += 0.05
        score = min(score, 1.0)
        if score >= 0.7:
            category = "critical"
        elif score >= 0.4:
            category = "high"
        elif score >= 0.2:
            category = "moderate"
        else:
            category = "low"
        factors: list[str] = []
        if acwr > 1.3:
            factors.append(f"high ACWR ({acwr:.2f})")
        if recent_sprints > 25:
            factors.append(f"high sprint volume ({recent_sprints})")
        if fatigue_index > 15:
            factors.append(f"elevated fatigue index ({fatigue_index:.1f})")
        if days_since_rest > 10:
            factors.append(f"long without rest ({days_since_rest} days)")
        return {
            "risk_score": round(score, 3),
            "risk_category": category,
            "key_risk_factors": factors if factors else ["no significant risk factors detected"],
        }

    def compute_recovery_recommendation(self, risk_score: float, player_position: str) -> str:
        if risk_score >= 0.7:
            risk_level = "critical"
        elif risk_score >= 0.4:
            risk_level = "high"
        elif risk_score >= 0.2:
            risk_level = "moderate"
        else:
            risk_level = "low"
        pos_recs = RECOVERY_RECOMMENDATIONS.get(player_position, DEFAULT_RECOMMENDATION)
        return pos_recs.get(risk_level, DEFAULT_RECOMMENDATION[risk_level])
