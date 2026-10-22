"""Model calibration metrics — xG reliability, Brier score, log loss."""

from __future__ import annotations

import math
from typing import Any


class ModelCalibrator:
    def compute_calibration_stats(self, events: list[dict]) -> dict[str, Any]:
        total_xg = 0.0
        actual_goals = 0
        brier_sum = 0.0
        n_shots = 0

        for event in events:
            if event.get("type") != "shot":
                continue
            xg = event.get("xg", event.get("metadata", {}).get("xg", 0.0))
            is_goal = event.get("is_goal", False)
            total_xg += xg
            if is_goal:
                actual_goals += 1
            brier_sum += (is_goal - xg) ** 2
            n_shots += 1

        calibration_error = abs(total_xg - actual_goals)
        brier_score = brier_sum / n_shots if n_shots > 0 else 0.0

        return {
            "total_xg": round(total_xg, 4),
            "actual_goals": actual_goals,
            "calibration_error": round(calibration_error, 4),
            "brier_score": round(brier_score, 4),
            "n_shots": n_shots,
        }

    def compute_reliability_curve(self, events: list[dict], n_bins: int = 10) -> list[dict]:
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]

        for event in events:
            if event.get("type") != "shot":
                continue
            xg = event.get("xg", event.get("metadata", {}).get("xg", 0.0))
            is_goal = event.get("is_goal", False)
            bin_idx = min(n_bins - 1, int(xg * n_bins))
            bins[bin_idx].append((xg, is_goal))

        curve: list[dict[str, Any]] = []
        for i, group in enumerate(bins):
            if not group:
                bin_start = i / n_bins
                bin_end = (i + 1) / n_bins
                curve.append(
                    {
                        "bin_range": f"{bin_start:.1f}-{bin_end:.1f}",
                        "expected_rate": round((i + 0.5) / n_bins, 3),
                        "observed_rate": None,
                        "count": 0,
                    }
                )
                continue
            expected = sum(xg for xg, _ in group) / len(group)
            observed = sum(1 for _, g in group if bool(g)) / len(group)
            bin_start = i / n_bins
            bin_end = (i + 1) / n_bins
            curve.append(
                {
                    "bin_range": f"{bin_start:.1f}-{bin_end:.1f}",
                    "expected_rate": round(expected, 3),
                    "observed_rate": round(observed, 3),
                    "count": len(group),
                }
            )

        return curve

    def compute_log_loss(self, events: list[dict]) -> dict[str, Any]:
        nll = 0.0
        n_shots = 0

        for event in events:
            if event.get("type") != "shot":
                continue
            xg = event.get("xg", event.get("metadata", {}).get("xg", 0.0))
            is_goal = event.get("is_goal", False)
            p = max(min(xg, 1.0 - 1e-15), 1e-15)
            nll += is_goal * math.log(p) + (1 - is_goal) * math.log(1 - p)
            n_shots += 1

        log_loss_val = -nll / n_shots if n_shots > 0 else 0.0

        return {
            "log_loss": round(log_loss_val, 4),
            "n_shots": n_shots,
        }

    def generate_calibration_report(self, events: list[dict]) -> dict[str, Any]:
        stats = self.compute_calibration_stats(events)
        reliability_curve = self.compute_reliability_curve(events)
        log_loss_result = self.compute_log_loss(events)

        if stats["n_shots"] == 0:
            return {
                **stats,
                **log_loss_result,
                "reliability_curve": reliability_curve,
                "status": "insufficient_data",
            }

        brier = stats["brier_score"]
        cal_error = stats["calibration_error"]
        total_xg = stats["total_xg"]
        actual = stats["actual_goals"]

        if cal_error <= 0.5 and brier <= 0.2:
            status = "well_calibrated"
        elif actual > total_xg:
            status = "underconfident"
        else:
            status = "overconfident"

        return {
            **stats,
            **log_loss_result,
            "reliability_curve": reliability_curve,
            "status": status,
        }
