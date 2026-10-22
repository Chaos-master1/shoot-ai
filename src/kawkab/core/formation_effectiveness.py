"""Formation Effectiveness vs Opponents.

Analyzes how a team's formation performs against specific opponent
formations and computes tactical flexibility scores. All numpy-only.
"""

from __future__ import annotations

from typing import Any


class FormationEffectivenessAnalyzer:
    def analyze_vs_formation(
        self, formations_data: dict[str, Any], opponent_formation: str
    ) -> dict[str, Any]:
        matches = formations_data.get("matches", [])
        relevant = [
            m
            for m in matches
            if m.get("opponent_formation", "").strip() == opponent_formation.strip()
        ]
        n = len(relevant)
        if n == 0:
            return {
                "opponent_formation": opponent_formation,
                "matches_analyzed": 0,
                "avg_goals_scored": 0.0,
                "avg_goals_conceded": 0.0,
                "avg_xg_for": 0.0,
                "avg_xg_against": 0.0,
                "avg_possession": 0.0,
                "avg_pass_completion": 0.0,
                "avg_chances_created": 0.0,
                "avg_pressing_intensity": 0.0,
                "win_rate": 0.0,
            }
        totals: dict[str, float] = {
            "goals_scored": 0,
            "goals_conceded": 0,
            "xg_for": 0,
            "xg_against": 0,
            "possession": 0,
            "pass_completion": 0,
            "chances_created": 0,
            "pressing_intensity": 0,
        }
        wins = 0
        for m in relevant:
            totals["goals_scored"] += m.get("goals_scored", 0)
            totals["goals_conceded"] += m.get("goals_conceded", 0)
            totals["xg_for"] += m.get("xg_for", 0.0)
            totals["xg_against"] += m.get("xg_against", 0.0)
            totals["possession"] += m.get("possession", 50.0)
            totals["pass_completion"] += m.get("pass_completion", 80.0)
            totals["chances_created"] += m.get("chances_created", 0)
            totals["pressing_intensity"] += m.get("pressing_intensity", 10.0)
            if m.get("goals_scored", 0) > m.get("goals_conceded", 0):
                wins += 1
        return {
            "opponent_formation": opponent_formation,
            "matches_analyzed": n,
            "avg_goals_scored": round(totals["goals_scored"] / n, 2),
            "avg_goals_conceded": round(totals["goals_conceded"] / n, 2),
            "avg_xg_for": round(totals["xg_for"] / n, 3),
            "avg_xg_against": round(totals["xg_against"] / n, 3),
            "avg_possession": round(totals["possession"] / n, 1),
            "avg_pass_completion": round(totals["pass_completion"] / n, 1),
            "avg_chances_created": round(totals["chances_created"] / n, 1),
            "avg_pressing_intensity": round(totals["pressing_intensity"] / n, 1),
            "win_rate": round(wins / n * 100, 1) if n else 0.0,
        }

    def compare_formation_performances(
        self, formation_history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not formation_history:
            return {
                "best_formation": "",
                "worst_formation": "",
                "formation_stats": {},
                "best_vs_opponent": {},
            }
        formation_stats: dict[str, dict[str, float]] = {}
        for m in formation_history:
            fm = m.get("formation", "")
            if not fm:
                continue
            if fm not in formation_stats:
                formation_stats[fm] = {
                    "matches": 0,
                    "goals_scored": 0,
                    "goals_conceded": 0,
                    "xg_for": 0.0,
                    "xg_against": 0.0,
                    "possession": 0.0,
                    "pass_completion": 0.0,
                    "wins": 0,
                }
            formation_stats[fm]["matches"] += 1
            formation_stats[fm]["goals_scored"] += m.get("goals_scored", 0)
            formation_stats[fm]["goals_conceded"] += m.get("goals_conceded", 0)
            formation_stats[fm]["xg_for"] += m.get("xg_for", 0.0)
            formation_stats[fm]["xg_against"] += m.get("xg_against", 0.0)
            formation_stats[fm]["possession"] += m.get("possession", 50.0)
            formation_stats[fm]["pass_completion"] += m.get("pass_completion", 80.0)
            if m.get("goals_scored", 0) > m.get("goals_conceded", 0):
                formation_stats[fm]["wins"] += 1
        processed: dict[str, dict[str, float]] = {}
        for fm, stats in formation_stats.items():
            n = stats["matches"]
            processed[fm] = {
                "matches_played": n,
                "avg_goals_scored": round(stats["goals_scored"] / n, 2),
                "avg_goals_conceded": round(stats["goals_conceded"] / n, 2),
                "goal_diff": round((stats["goals_scored"] - stats["goals_conceded"]) / n, 2),
                "avg_xg_for": round(stats["xg_for"] / n, 3),
                "avg_xg_against": round(stats["xg_against"] / n, 3),
                "avg_possession": round(stats["possession"] / n, 1),
                "avg_pass_completion": round(stats["pass_completion"] / n, 1),
                "win_rate": round(stats["wins"] / n * 100, 1),
            }
        sorted_fm = sorted(processed.items(), key=lambda x: x[1]["goal_diff"], reverse=True)
        best = sorted_fm[0][0] if sorted_fm else ""
        worst = sorted_fm[-1][0] if sorted_fm else ""
        return {
            "best_formation": best,
            "worst_formation": worst,
            "formation_stats": processed,
            "best_vs_opponent": {},
        }

    def compute_formation_flexibility_score(
        self, formation_history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not formation_history:
            return {
                "flexibility_score": 0.0,
                "formations_used": 0,
                "total_matches": 0,
                "verdict": "No data",
            }
        formations = {m.get("formation", "") for m in formation_history if m.get("formation")}
        n_formations = len(formations)
        n_matches = len(formation_history)
        if n_formations == 0:
            return {
                "flexibility_score": 0.0,
                "formations_used": 0,
                "total_matches": 0,
                "verdict": "No formations recorded",
            }
        _ = min(n_formations, 5)
        formation_ratio = n_formations / 5.0
        success_sum = 0.0
        n_success = 0
        for fm in formations:
            fm_matches = [m for m in formation_history if m.get("formation") == fm]
            if not fm_matches:
                continue
            wins = sum(
                1 for m in fm_matches if m.get("goals_scored", 0) > m.get("goals_conceded", 0)
            )
            win_rate = wins / len(fm_matches)
            success_sum += win_rate
            n_success += 1
        avg_win_rate = success_sum / n_success if n_success else 0.0
        flexibility = min(formation_ratio * 0.5 + avg_win_rate * 0.5, 1.0)
        if flexibility > 0.7:
            verdict = "Highly flexible - team adapts well to different formations"
        elif flexibility > 0.4:
            verdict = "Moderately flexible - team has a preferred system but can adapt"
        else:
            verdict = "Rigid - team struggles when changing formation"
        return {
            "flexibility_score": round(flexibility, 2),
            "formations_used": n_formations,
            "total_matches": n_matches,
            "formation_list": sorted(formations),
            "verdict": verdict,
        }
