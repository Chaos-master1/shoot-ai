"""Player stat benchmarking — within-squad and positional percentiles."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

POSITION_GROUPS: dict[str, list[str]] = {
    "CB": ["CB", "LCB", "RCB"],
    "FB": ["LB", "RB", "LWB", "RWB"],
    "CM": ["CM", "CDM", "CAM"],
    "Winger": ["LW", "RW", "LM", "RM"],
    "ST": ["ST", "CF"],
}


def get_position_groups() -> list[str]:
    return list(POSITION_GROUPS.keys())


@dataclass
class PercentileResult:
    stat_name: str
    value: float
    percentile: float
    squad_min: float
    squad_max: float
    squad_mean: float
    z_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stat_name": self.stat_name,
            "value": round(self.value, 2),
            "percentile": round(self.percentile, 1),
            "squad_min": round(self.squad_min, 2),
            "squad_max": round(self.squad_max, 2),
            "squad_mean": round(self.squad_mean, 2),
            "z_score": round(self.z_score, 2),
        }


@dataclass
class PlayerBenchmark:
    track_id: int
    name: str = ""
    position: str = ""
    results: list[PercentileResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "position": self.position,
            "results": [r.to_dict() for r in self.results],
        }


def _percentile(values: list[float], value: float) -> float:
    """Compute percentile rank of value within list."""
    if not values:
        return 50.0
    n_below = sum(1 for v in values if v <= value)
    return (n_below / len(values)) * 100.0


def _z_score(values: list[float], value: float) -> float:
    """Compute z-score of value within list."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    if std < 1e-9:
        return 0.0
    return (value - mean) / std


def _group_positions(position_group: str) -> set[str]:
    """Resolve position group to set of position labels (upper case)."""
    raw = POSITION_GROUPS.get(position_group)
    if raw is None:
        return set()
    return {p.upper() for p in raw}


def compute_player_benchmarks(
    player_ratings: dict[int, dict[str, Any]],
    position_group: str | None = None,
) -> list[PlayerBenchmark]:
    """Compute within-squad percentile benchmarks for each player.

    Compares each player against the squad for key metrics.
    When *position_group* is provided, percentiles are computed only
    against players whose position falls within that group.

    Args:
        player_ratings: Dict mapping track_id to rating dict with numeric fields.
        position_group: Optional group name (CB, FB, CM, Winger, ST) to filter by.

    Returns:
        List of PlayerBenchmark per player (filtered by group if given).
    """
    if not player_ratings:
        return []

    # Filter by position group if requested
    if position_group is not None:
        group_positions = _group_positions(position_group)
        if group_positions:
            filtered = {}
            for tid, pdata in player_ratings.items():
                if isinstance(pdata, dict):
                    pos = (pdata.get("position") or "").upper().strip()
                    if pos in group_positions:
                        filtered[tid] = pdata
            player_ratings = filtered
            if not player_ratings:
                return []

    # Collect all stat values per field across players
    stat_fields = [
        "pass_accuracy",
        "shots",
        "tackles",
        "distance_covered_m",
        "passes_attempted",
        "passes_completed",
    ]
    stat_values: dict[str, list[float]] = {f: [] for f in stat_fields}
    stat_names = {
        "pass_accuracy": "Pass Accuracy",
        "shots": "Shots",
        "tackles": "Tackles",
        "distance_covered_m": "Distance (m)",
        "passes_attempted": "Passes Attempted",
        "passes_completed": "Passes Completed",
    }

    # Extract numeric fields from player rating data
    for _tid, pdata in player_ratings.items():
        if not isinstance(pdata, dict):
            continue
        for stat_field in stat_fields:
            val = pdata.get(stat_field)
            if isinstance(val, (int, float)):
                stat_values[stat_field].append(float(val))

    results: list[PlayerBenchmark] = []
    for tid, pdata in player_ratings.items():
        if not isinstance(pdata, dict):
            continue
        benchmarks: list[PercentileResult] = []
        for stat_field in stat_fields:
            val = pdata.get(stat_field)
            if not isinstance(val, (int, float)):
                continue
            vals = stat_values[stat_field]
            results_list = PercentileResult(
                stat_name=stat_names.get(stat_field, stat_field),
                value=float(val),
                percentile=_percentile(vals, float(val)),
                squad_min=min(vals) if vals else 0.0,
                squad_max=max(vals) if vals else 0.0,
                squad_mean=sum(vals) / len(vals) if vals else 0.0,
                z_score=_z_score(vals, float(val)),
            )
            benchmarks.append(results_list)

        name = pdata.get("name", f"Player #{tid}")
        position = pdata.get("position", "unknown")
        results.append(
            PlayerBenchmark(
                track_id=tid,
                name=name,
                position=position,
                results=benchmarks,
            )
        )

    return results
