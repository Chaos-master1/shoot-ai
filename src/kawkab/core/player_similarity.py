"""Player Similarity Engine — compare player profiles using cosine similarity.

All numpy-only, no pandas/scipy/sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

STAT_NAMES: list[str] = [
    "pass_completion_pct",
    "passes_per_90",
    "key_passes_per_90",
    "xA_per_90",
    "xG_per_90",
    "shots_per_90",
    "progressive_passes_per_90",
    "carries_per_90",
    "tackles_per_90",
    "interceptions_per_90",
    "pressures_per_90",
    "aerial_win_pct",
    "distance_per_90",
    "speed_max",
    "pass_completion_short",
    "pass_completion_medium",
    "pass_completion_long",
]

STAT_DIM = len(STAT_NAMES)

STAT_MEANS: dict[str, float] = {
    "pass_completion_pct": 78.0,
    "passes_per_90": 45.0,
    "key_passes_per_90": 1.5,
    "xA_per_90": 0.15,
    "xG_per_90": 0.12,
    "shots_per_90": 1.8,
    "progressive_passes_per_90": 4.0,
    "carries_per_90": 8.0,
    "tackles_per_90": 2.5,
    "interceptions_per_90": 1.8,
    "pressures_per_90": 12.0,
    "aerial_win_pct": 50.0,
    "distance_per_90": 10.0,
    "speed_max": 30.0,
    "pass_completion_short": 88.0,
    "pass_completion_medium": 78.0,
    "pass_completion_long": 55.0,
}

STAT_STDS: dict[str, float] = {
    "pass_completion_pct": 8.0,
    "passes_per_90": 12.0,
    "key_passes_per_90": 0.8,
    "xA_per_90": 0.10,
    "xG_per_90": 0.10,
    "shots_per_90": 1.0,
    "progressive_passes_per_90": 2.5,
    "carries_per_90": 4.0,
    "tackles_per_90": 1.2,
    "interceptions_per_90": 0.9,
    "pressures_per_90": 5.0,
    "aerial_win_pct": 15.0,
    "distance_per_90": 2.0,
    "speed_max": 3.0,
    "pass_completion_short": 5.0,
    "pass_completion_medium": 8.0,
    "pass_completion_long": 12.0,
}

# Position archetypes: average z-score profiles for each position group
POSITION_ARCHETYPES: dict[str, list[float]] = {
    "GK": [
        -1.5,
        -1.8,
        -1.5,
        -1.2,
        -1.0,
        -1.5,
        -1.5,
        -1.5,
        -1.0,
        -1.0,
        -1.0,
        0.8,
        -0.5,
        -0.5,
        -1.5,
        -1.5,
        -1.5,
    ],
    "CB": [
        0.5,
        0.0,
        -0.8,
        -0.5,
        -0.8,
        -1.0,
        -0.5,
        -0.8,
        1.2,
        1.5,
        1.0,
        1.5,
        0.0,
        -0.5,
        1.0,
        0.5,
        0.0,
    ],
    "FB": [
        0.5,
        0.5,
        0.2,
        0.0,
        -0.3,
        -0.2,
        0.5,
        1.5,
        1.0,
        0.8,
        0.8,
        0.0,
        1.0,
        0.8,
        0.5,
        0.0,
        -0.5,
    ],
    "DM": [
        1.0,
        1.0,
        0.0,
        0.0,
        -0.5,
        -0.5,
        0.0,
        0.0,
        1.5,
        1.2,
        1.2,
        0.5,
        0.5,
        0.0,
        1.0,
        1.0,
        0.0,
    ],
    "CM": [
        1.0,
        1.5,
        0.8,
        0.5,
        0.3,
        0.5,
        0.5,
        0.5,
        0.5,
        0.3,
        0.5,
        -0.5,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ],
    "AM": [
        0.5,
        0.5,
        1.5,
        1.5,
        1.0,
        1.2,
        0.8,
        0.5,
        -0.5,
        -0.5,
        -0.2,
        -1.0,
        0.5,
        0.0,
        0.0,
        0.5,
        -0.5,
    ],
    "WINGER": [
        -0.5,
        -0.5,
        0.8,
        0.5,
        1.0,
        1.5,
        0.8,
        1.2,
        -0.5,
        -0.5,
        0.0,
        -1.2,
        0.5,
        1.5,
        -0.5,
        0.0,
        -0.5,
    ],
    "ST": [
        -1.0,
        -1.0,
        -0.5,
        0.0,
        2.0,
        2.5,
        0.0,
        0.0,
        -1.0,
        -1.0,
        -0.5,
        0.5,
        -0.5,
        0.5,
        -1.0,
        -0.5,
        -0.5,
    ],
}


def _z_score(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return (value - mean) / std


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


@dataclass
class PlayerProfile:
    player_id: Any
    vector: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"player_id": self.player_id, "vector": self.vector}


@dataclass
class SimilarityResult:
    player_id: Any
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {"player_id": self.player_id, "similarity": round(self.similarity, 4)}


@dataclass
class ComparisonResult:
    player_a_id: Any
    player_b_id: Any
    overall_similarity: float
    per_stat: list[dict[str, Any]]
    strengths_a: list[str]
    strengths_b: list[str]
    weaknesses_a: list[str]
    weaknesses_b: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_a_id": self.player_a_id,
            "player_b_id": self.player_b_id,
            "overall_similarity": round(self.overall_similarity, 4),
            "per_stat": self.per_stat,
            "strengths_a": self.strengths_a,
            "strengths_b": self.strengths_b,
            "weaknesses_a": self.weaknesses_a,
            "weaknesses_b": self.weaknesses_b,
        }


class PlayerSimilarityEngine:
    """Build player profiles and find similar players via cosine similarity."""

    def build_player_profile(self, player_stats: dict[str, float]) -> PlayerProfile:
        player_id = player_stats.get("player_id", 0)
        vector = []
        for stat in STAT_NAMES:
            raw = player_stats.get(stat, STAT_MEANS[stat])
            z = _z_score(raw, STAT_MEANS[stat], STAT_STDS[stat])
            vector.append(round(z, 4))
        return PlayerProfile(player_id=player_id, vector=vector)

    def find_similar_players(
        self,
        target_player: PlayerProfile,
        player_pool: list[PlayerProfile],
        top_n: int = 5,
    ) -> list[SimilarityResult]:
        if not player_pool:
            return []

        target_vec = np.array(target_player.vector, dtype=np.float64)
        scored: list[SimilarityResult] = []

        for profile in player_pool:
            if profile.player_id == target_player.player_id:
                continue
            profile_vec = np.array(profile.vector, dtype=np.float64)
            sim = _cosine_similarity(target_vec, profile_vec)
            scored.append(SimilarityResult(player_id=profile.player_id, similarity=sim))

        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:top_n]

    def compare_players(
        self,
        player_a_profile: PlayerProfile,
        player_b_profile: PlayerProfile,
    ) -> ComparisonResult:
        vec_a = np.array(player_a_profile.vector, dtype=np.float64)
        vec_b = np.array(player_b_profile.vector, dtype=np.float64)
        overall_sim = _cosine_similarity(vec_a, vec_b)

        per_stat: list[dict[str, Any]] = []

        for i, stat in enumerate(STAT_NAMES):
            diff = player_a_profile.vector[i] - player_b_profile.vector[i]
            better = "a" if diff > 0 else ("b" if diff < 0 else "equal")
            per_stat.append(
                {
                    "stat": stat,
                    "a_value": player_a_profile.vector[i],
                    "b_value": player_b_profile.vector[i],
                    "diff": round(diff, 4),
                    "better": better,
                }
            )

        strengths_a = [s["stat"] for s in per_stat if s["better"] == "a" and abs(s["diff"]) > 0.3]
        strengths_b = [s["stat"] for s in per_stat if s["better"] == "b" and abs(s["diff"]) > 0.3]
        weaknesses_a = strengths_b[:]
        weaknesses_b = strengths_a[:]

        return ComparisonResult(
            player_a_id=player_a_profile.player_id,
            player_b_id=player_b_profile.player_id,
            overall_similarity=overall_sim,
            per_stat=per_stat,
            strengths_a=strengths_a[:5],
            strengths_b=strengths_b[:5],
            weaknesses_a=weaknesses_a[:5],
            weaknesses_b=weaknesses_b[:5],
        )

    def compute_position_similarity(
        self,
        player_profile: PlayerProfile,
        position_archetype_profile: list[float],
    ) -> float:
        vec_a = np.array(player_profile.vector, dtype=np.float64)
        vec_b = np.array(position_archetype_profile[:STAT_DIM], dtype=np.float64)
        return _cosine_similarity(vec_a, vec_b)
