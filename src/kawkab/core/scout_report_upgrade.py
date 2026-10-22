"""Enhanced scout report generation with video clip integration.

Produces rich scouting reports with strengths, weaknesses,
statistical percentiles, video evidence links, and similar players.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class ScoutStatPercentile:
    stat_name: str = ""
    value: float = 0.0
    percentile: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stat": self.stat_name,
            "value": round(self.value, 2),
            "percentile": round(self.percentile, 1),
            "desc": self.description,
        }


@dataclass
class ScoutStrengthWeakness:
    category: str = ""
    detail: str = ""
    evidence_stat: str = ""
    evidence_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "detail": self.detail,
            "evidence_stat": self.evidence_stat,
            "evidence_value": round(self.evidence_value, 2),
        }


@dataclass
class ScoutVideoClip:
    clip_id: str = ""
    timestamp: float = 0.0
    duration_s: float = 5.0
    label: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "label": self.label,
            "tags": self.tags,
        }


@dataclass
class ScoutReport:
    player_name: str = ""
    player_id: int = 0
    position: str = ""
    strengths: list[ScoutStrengthWeakness] = field(default_factory=list)
    weaknesses: list[ScoutStrengthWeakness] = field(default_factory=list)
    percentiles: list[ScoutStatPercentile] = field(default_factory=list)
    video_clips: list[ScoutVideoClip] = field(default_factory=list)
    similar_players: list[dict[str, Any]] = field(default_factory=list)
    overall_rating: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "player_id": self.player_id,
            "position": self.position,
            "strengths": [s.to_dict() for s in self.strengths],
            "weaknesses": [w.to_dict() for w in self.weaknesses],
            "percentiles": [p.to_dict() for p in self.percentiles],
            "video_clips": [c.to_dict() for c in self.video_clips],
            "similar_players": self.similar_players,
            "overall_rating": round(self.overall_rating, 1),
        }


def _compute_stats(
    player_events: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute summary statistics from player events."""
    passes = [e for e in player_events if e.get("type") == "pass"]
    shots = [e for e in player_events if e.get("type") in ("shot", "goal")]
    tackles = [e for e in player_events if e.get("type") == "tackle"]
    interceptions = [e for e in player_events if e.get("type") == "interception"]
    crosses = [e for e in player_events if e.get("type") == "cross"]

    completed_passes = sum(1 for p in passes if p.get("completed", True))
    total_passes = len(passes)
    pass_acc = (completed_passes / max(total_passes, 1)) * 100.0

    shot_xg = sum(e.get("xG", 0.0) for e in shots)

    progressive_passes = sum(1 for p in passes if p.get("end_x", 0) - p.get("start_x", 0) > 10)

    return {
        "passes": float(total_passes),
        "pass_accuracy": pass_acc,
        "shots": float(len(shots)),
        "xg_total": shot_xg,
        "tackles": float(len(tackles)),
        "interceptions": float(len(interceptions)),
        "crosses": float(len(crosses)),
        "progressive_passes": float(progressive_passes),
        "total_events": float(len(player_events)),
    }


def _estimate_percentile(
    value: float,
    stat_name: str,
    position: str,
) -> float:
    """Estimate percentile rank based on position-specific benchmarks.

    Uses heuristic benchmarks per position for each stat type.
    Args:
        value: The player's value for this stat.
        stat_name: Name of the stat.
        position: Player's role/position.

    Returns:
        Estimated percentile (0-100).
    """
    benchmarks: dict[str, dict[str, tuple[float, float]]] = {
        "pass_accuracy": {
            "CB": (75, 92),
            "FB": (72, 90),
            "DM": (73, 91),
            "MF": (70, 90),
            "AM": (68, 88),
            "W": (65, 85),
            "FW": (62, 82),
        },
        "tackles": {
            "CB": (1.0, 4.0),
            "FB": (1.0, 3.5),
            "DM": (1.5, 4.5),
            "MF": (0.5, 2.5),
            "AM": (0.2, 1.5),
            "W": (0.3, 1.5),
            "FW": (0.1, 1.0),
        },
        "interceptions": {
            "CB": (0.5, 3.0),
            "FB": (0.5, 2.5),
            "DM": (1.0, 3.5),
            "MF": (0.3, 2.0),
            "AM": (0.1, 1.2),
            "W": (0.2, 1.0),
            "FW": (0.1, 0.8),
        },
        "crosses": {
            "FB": (0.5, 4.0),
            "W": (1.0, 6.0),
            "MF": (0.2, 2.0),
            "AM": (0.1, 1.5),
            "FW": (0.1, 1.0),
        },
        "progressive_passes": {
            "CB": (1.0, 6.0),
            "FB": (2.0, 8.0),
            "DM": (3.0, 10.0),
            "MF": (4.0, 12.0),
            "AM": (3.0, 10.0),
            "W": (2.0, 8.0),
            "FW": (1.0, 5.0),
        },
        "shots": {
            "CB": (0.0, 0.5),
            "FB": (0.0, 1.0),
            "DM": (0.1, 1.5),
            "MF": (0.5, 3.0),
            "AM": (1.0, 5.0),
            "W": (0.5, 3.0),
            "FW": (1.0, 6.0),
        },
    }

    if stat_name not in benchmarks:
        return 50.0

    pos_map: dict[str, str] = {
        "centre_back": "CB",
        "full_back": "FB",
        "inverted_fullback": "FB",
        "defensive_midfielder": "DM",
        "box_to_box_midfielder": "MF",
        "wide_midfielder": "MF",
        "attacking_midfielder": "AM",
        "winger": "W",
        "inside_forward": "W",
        "target_forward": "FW",
        "false_nine": "FW",
        "poacher": "FW",
        "wide_playmaker": "W",
        "utility_player": "MF",
    }
    pos_key = pos_map.get(position, "MF")
    pos_benchmarks = benchmarks.get(stat_name, {})

    if pos_key not in pos_benchmarks:
        avg_keys = [k for k in pos_benchmarks if k != "FW"]
        if avg_keys:
            low = min(v[0] for v in pos_benchmarks.values())
            high = max(v[1] for v in pos_benchmarks.values())
        else:
            return 50.0
    else:
        low, high = pos_benchmarks[pos_key]

    if high <= low:
        return 50.0
    clamped = max(low, min(value, high))
    return ((clamped - low) / (high - low)) * 100.0


def generate_scout_report(
    player_events: list[dict[str, Any]],
    player_profile: dict[str, Any],
    video_clips: list[dict[str, Any]] | None = None,
) -> ScoutReport:
    """Generate a comprehensive scout report.

    Args:
        player_events: List of event dicts for the player.
        player_profile: Dict with name, id, position fields.
        video_clips: Optional list of clip dicts with id, timestamp,
            duration, label, tags.

    Returns:
        ScoutReport with all sections populated.
    """
    stats = _compute_stats(player_events)
    position = player_profile.get("position", "unknown")
    player_name = player_profile.get("name", f"Player {player_profile.get('id', 0)}")
    player_id = player_profile.get("id", 0)

    # percentiles
    percentiles: list[ScoutStatPercentile] = []
    for stat_name in (
        "pass_accuracy",
        "tackles",
        "interceptions",
        "crosses",
        "progressive_passes",
        "shots",
    ):
        if stat_name not in stats:
            continue
        pct = _estimate_percentile(stats[stat_name], stat_name, position)
        desc_map = {
            "pass_accuracy": "Pass completion rate",
            "tackles": "Tackles per match",
            "interceptions": "Interceptions per match",
            "crosses": "Crosses per match",
            "progressive_passes": "Progressive passes per match",
            "shots": "Shots per match",
        }
        percentiles.append(
            ScoutStatPercentile(
                stat_name=stat_name,
                value=stats[stat_name],
                percentile=round(pct, 1),
                description=desc_map.get(stat_name, stat_name),
            )
        )

    # strengths (top 3 percentiles)
    sorted_pcts = sorted(percentiles, key=lambda p: p.percentile, reverse=True)
    strengths: list[ScoutStrengthWeakness] = []
    for p in sorted_pcts[:3]:
        strengths.append(
            ScoutStrengthWeakness(
                category="Strength",
                detail=f"Exceptional {p.description.lower()} ({p.percentile:.0f}th percentile)",
                evidence_stat=p.stat_name,
                evidence_value=p.value,
            )
        )

    # weaknesses (bottom 3 percentiles)
    weaknesses: list[ScoutStrengthWeakness] = []
    for p in sorted_pcts[-3:]:
        if p.percentile < 50:
            weaknesses.append(
                ScoutStrengthWeakness(
                    category="Weakness",
                    detail=f"Below-average {p.description.lower()} ({p.percentile:.0f}th percentile)",
                    evidence_stat=p.stat_name,
                    evidence_value=p.value,
                )
            )

    # video clips
    clips: list[ScoutVideoClip] = []
    if video_clips:
        for clip in video_clips:
            clips.append(
                ScoutVideoClip(
                    clip_id=clip.get("id", ""),
                    timestamp=clip.get("timestamp", 0.0),
                    duration_s=clip.get("duration_s", 5.0),
                    label=clip.get("label", ""),
                    tags=clip.get("tags", []),
                )
            )

    # similar players (heuristic based on stat profile)
    similar_players: list[dict[str, Any]] = []
    profile_vec = [
        stats.get(s, 0.0) for s in ("pass_accuracy", "tackles", "shots", "progressive_passes")
    ]
    bench_players = [
        ("Kevin De Bruyne", "AM", 92, "creative playmaker"),
        ("Rodri", "DM", 90, "defensive controller"),
        ("Erling Haaland", "FW", 95, "elite finisher"),
        ("Trent Alexander-Arnold", "FB", 88, "wide creator"),
        ("Jude Bellingham", "MF", 91, "box-to-box dynamo"),
    ]
    for bname, bpos, brating, bdesc in bench_players:
        bvec = {
            "CB": (86, 3.0, 0.2, 3.0),
            "FB": (80, 2.5, 0.5, 5.0),
            "DM": (88, 4.0, 0.3, 6.0),
            "MF": (84, 2.0, 1.5, 8.0),
            "AM": (82, 1.0, 3.0, 6.0),
            "W": (78, 1.2, 2.5, 4.0),
            "FW": (75, 0.5, 4.0, 2.0),
        }
        bv = bvec.get(bpos, (80, 1.0, 1.0, 4.0))
        sim = (
            1.0
            - sum(abs(a - b) / max(abs(b), 1.0) for a, b in zip(profile_vec, bv, strict=False))
            / 4.0
        )
        if sim > 0.3:
            similar_players.append(
                {
                    "name": bname,
                    "position": bpos,
                    "rating": brating,
                    "description": bdesc,
                    "similarity": round(max(0.0, sim * 100), 1),
                }
            )

    similar_players.sort(key=lambda x: x["similarity"], reverse=True)

    # overall rating
    avg_pct = sum(p.percentile for p in percentiles) / max(len(percentiles), 1)
    overall_rating = 30.0 + avg_pct * 0.7

    return ScoutReport(
        player_name=player_name,
        player_id=player_id,
        position=position,
        strengths=strengths,
        weaknesses=weaknesses,
        percentiles=percentiles,
        video_clips=clips,
        similar_players=similar_players[:3],
        overall_rating=round(overall_rating, 1),
    )
