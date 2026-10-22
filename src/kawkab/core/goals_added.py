"""Goals Added (g+) — total on-ball contribution expressed in goals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GoalsAddedReport:
    player_id: str
    total_g_plus: float
    g_plus_per_90: float
    components: dict
    per_game: list[dict]
    percentile_vs_position: float


XG_WEIGHT = 1.0
XA_WEIGHT = 0.8
XT_WEIGHT = 0.3
DEF_WEIGHT = 0.02
OBV_WEIGHT = 1.0


def _compute_g_plus_from_match(m: dict) -> float:
    xg = m.get("xg", 0.0) * XG_WEIGHT
    xa = m.get("xa", 0.0) * XA_WEIGHT
    xt = m.get("xt", 0.0) * XT_WEIGHT
    defensive = m.get("defensive_actions", 0) * DEF_WEIGHT
    obv = m.get("obv", 0.0) * OBV_WEIGHT
    return xg + xa + xt + defensive + obv


def _percentile_value(value: float, distribution: list[float]) -> float:
    if not distribution:
        return 50.0
    count_below = sum(1 for v in distribution if v < value)
    return round(count_below / len(distribution) * 100.0, 1)


def compute_goals_added(
    player_id: str,
    match_stats: list[dict],
    position: str,
    minutes_played: int = 90,
    league_stats: dict | None = None,
) -> GoalsAddedReport:
    total_xg = sum(m.get("xg", 0.0) for m in match_stats) * XG_WEIGHT
    total_xa = sum(m.get("xa", 0.0) for m in match_stats) * XA_WEIGHT
    total_xt = sum(m.get("xt", 0.0) for m in match_stats) * XT_WEIGHT
    total_def = sum(m.get("defensive_actions", 0) for m in match_stats) * DEF_WEIGHT
    total_obv = sum(m.get("obv", 0.0) for m in match_stats) * OBV_WEIGHT

    total_g_plus = total_xg + total_xa + total_xt + total_def + total_obv

    per_game: list[dict] = []
    for m in match_stats:
        g = _compute_g_plus_from_match(m)
        per_game.append(
            {
                "match_id": m.get("match_id", ""),
                "g_plus": round(g, 4),
                "minutes": m.get("minutes", 90),
            }
        )

    games = len(match_stats)
    total_minutes = sum(m.get("minutes", 90) for m in match_stats) if games > 0 else 0
    g_plus_per_90 = (total_g_plus / total_minutes * 90) if total_minutes > 0 else 0.0

    percentile = 50.0
    if league_stats:
        pos_key = position.lower()
        position_avgs = league_stats.get("position_averages", {})
        pos_g_plus = position_avgs.get(pos_key, {}).get("g_plus_per_90", [])
        if isinstance(pos_g_plus, list) and pos_g_plus:
            percentile = _percentile_value(g_plus_per_90, pos_g_plus)
        elif isinstance(pos_g_plus, (int, float)):
            pos_val = float(pos_g_plus)
            if g_plus_per_90 > pos_val:
                base = min(99.0, (g_plus_per_90 / pos_val) * 50.0 if pos_val > 0 else 50.0)
                percentile = min(99.0, 50.0 + base)
            else:
                base = min(50.0, (pos_val / g_plus_per_90) * 50.0 if g_plus_per_90 > 0 else 50.0)
                percentile = max(1.0, 50.0 - base)

    return GoalsAddedReport(
        player_id=player_id,
        total_g_plus=round(total_g_plus, 4),
        g_plus_per_90=round(g_plus_per_90, 4),
        components={
            "xg_contribution": round(total_xg, 4),
            "xa_contribution": round(total_xa, 4),
            "xt_contribution": round(total_xt, 4),
            "defensive_contribution": round(total_def, 4),
            "obv_contribution": round(total_obv, 4),
        },
        per_game=per_game,
        percentile_vs_position=round(percentile, 1),
    )
