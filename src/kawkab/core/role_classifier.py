"""Automated player role classification from event data.

Uses heuristic-based analysis of position heatmaps, pass direction,
shot locations, and defensive actions to classify player roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class PlayerRole:
    primary_role: str = ""
    secondary_role: str = ""
    confidence: float = 0.0
    role_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary_role,
            "secondary": self.secondary_role,
            "confidence": round(self.confidence, 2),
            "scores": {k: round(v, 2) for k, v in self.role_scores.items()},
        }


ROLE_DEFINITIONS: list[tuple[str, str, str]] = [
    ("goalkeeper", "Always near own goal, no shots, no attacking passes", "GK"),
    ("centre_back", "Deepest outfield position, few shots, mostly sideways/back passes", "CB"),
    ("full_back", "Wide positions in own half, some crosses, moderate defensive actions", "FB"),
    ("inverted_fullback", "Wide starting position but drifts centrally in attack", "IFB"),
    ("defensive_midfielder", "Central, in front of defense, many interceptions/tackles", "DM"),
    ("box_to_box_midfielder", "Covers both boxes, high work rate, shots + tackles", "BBM"),
    ("wide_midfielder", "Wide positions, crosses + dribbles", "WM"),
    ("attacking_midfielder", "Central attacking third, many shots, few defensive actions", "AM"),
    ("winger", "Wide and high, many crosses and dribbles, few defensive actions", "W"),
    ("inside_forward", "Starts wide but drifts central to shoot", "IF"),
    ("target_forward", "Central, high shot volume, aerial duels, receives long balls", "TF"),
    ("false_nine", "Drops deep from forward position, links play, fewer shots", "F9"),
    ("poacher", "Central, very high shot volume in box, few passes outside box", "POA"),
    ("wide_playmaker", "Wide but creates chances, high key passes, low dribbles", "WP"),
    ("utility_player", "Spread across multiple zones, no single strong profile", "UTIL"),
]


def _avg_position(
    events: list[dict[str, Any]], key_x: str = "start_x", key_y: str = "start_y"
) -> tuple[float, float]:
    xs = [e.get(key_x, 0) for e in events if e.get(key_x) is not None]
    ys = [e.get(key_y, 0) for e in events if e.get(key_y) is not None]
    if not xs or not ys:
        return (PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    return (float(np.mean(xs)), float(np.mean(ys)))


def classify_player_role(
    player_events: list[dict[str, Any]],
) -> PlayerRole:
    """Classify a player's role from their event data.

    Uses heuristic scoring across 15 role definitions based on
    position (avg x/y), pass direction bias, shot volume/location,
    defensive actions, and aerial duels.

    Args:
        player_events: List of event dicts for one player.

    Returns:
        PlayerRole with primary, secondary, and confidence.
    """
    if not player_events:
        return PlayerRole(primary_role="unknown", secondary_role="", confidence=0.0)

    avg_x, avg_y = _avg_position(player_events)
    avg_x_rel = avg_x / PITCH_LENGTH
    avg_y_rel = avg_y / PITCH_WIDTH

    passes = [e for e in player_events if e.get("type") == "pass"]
    shots = [e for e in player_events if e.get("type") in ("shot", "goal")]
    tackles = [e for e in player_events if e.get("type") == "tackle"]
    interceptions = [e for e in player_events if e.get("type") == "interception"]
    clearances = [e for e in player_events if e.get("type") == "clearance"]
    crosses = [e for e in player_events if e.get("type") == "cross"]

    def_actions = len(tackles) + len(interceptions) + len(clearances)
    total_actions = float(len(player_events))

    # pass direction bias
    forward_passes = 0
    backward_passes = 0
    lateral_passes = 0
    for p in passes:
        sx = p.get("start_x", 0)
        ex = p.get("end_x", 0)
        diff = ex - sx
        if diff > 5:
            forward_passes += 1
        elif diff < -5:
            backward_passes += 1
        else:
            lateral_passes += 1
    total_passes = len(passes)
    forward_pct = forward_passes / max(total_passes, 1)
    backward_pct = backward_passes / max(total_passes, 1)

    # shot characteristics
    _ = _avg_position(shots, "start_x", "start_y")[0] if shots else 0
    shot_volume = len(shots) / max(total_actions, 1)

    # wide vs central
    wide_pct = float(
        sum(
            1
            for e in player_events
            if e.get("start_y", PITCH_WIDTH / 2) < PITCH_WIDTH * 0.25
            or e.get("start_y", PITCH_WIDTH / 2) > PITCH_WIDTH * 0.75
        )
    )
    wide_pct /= max(total_actions, 1)

    scores: dict[str, float] = {}
    for role_name, _, _ in ROLE_DEFINITIONS:
        scores[role_name] = 0.0

    scores["goalkeeper"] = max(0, 1.0 - avg_x_rel * 3) * 10
    scores["goalkeeper"] += (1.0 if total_passes == 0 else 0) * 5

    scores["centre_back"] = max(0, 1.0 - avg_x_rel * 2) * 8
    scores["centre_back"] += (def_actions / max(total_actions, 1)) * 6
    scores["centre_back"] += backward_pct * 5

    fb_score = (1.0 - abs(avg_y_rel - 0.5) * 2) * 4
    fb_score += wide_pct * 6
    fb_score += max(0, 0.5 - avg_x_rel) * 4
    fb_score += len(crosses) / max(total_actions, 1) * 5
    scores["full_back"] = fb_score

    ifb_score = fb_score * 0.5
    _ = 1.0 - abs(avg_y_rel - 0.5) * 2
    ifb_score += max(0, avg_x_rel - 0.3) * 3
    scores["inverted_fullback"] = ifb_score

    dm_score = (1.0 - abs(avg_y_rel - 0.5) * 2) * 6
    dm_score += max(0, 0.55 - avg_x_rel) * 4
    dm_score += (def_actions / max(total_actions, 1)) * 7
    scores["defensive_midfielder"] = dm_score

    bbm_score = (1.0 - abs(avg_y_rel - 0.5) * 2) * 5
    bbm_score += shot_volume * 6
    bbm_score += (def_actions / max(total_actions, 1)) * 5
    bbm_score += forward_pct * 4
    scores["box_to_box_midfielder"] = bbm_score

    wm_score = float(wide_pct * 7)
    wm_score += len(crosses) / max(total_actions, 1) * 5
    wm_score += max(0, 0.5 - avg_x_rel) * 3
    scores["wide_midfielder"] = wm_score

    am_score = (1.0 - abs(avg_y_rel - 0.5) * 2) * 5
    am_score += max(0, avg_x_rel - 0.5) * 6
    am_score += shot_volume * 7
    am_score += forward_pct * 4
    am_score += (1.0 - def_actions / max(total_actions, 1)) * 3
    scores["attacking_midfielder"] = am_score

    winger_score = float(wide_pct * 8)
    winger_score += len(crosses) / max(total_actions, 1) * 6
    winger_score += max(0, avg_x_rel - 0.5) * 4
    winger_score += (1.0 - def_actions / max(total_actions, 1)) * 3
    scores["winger"] = winger_score

    iff_score = float(wide_pct * 4)
    iff_score += max(0, avg_x_rel - 0.6) * 6
    iff_score += shot_volume * 6
    iff_score += (1.0 - def_actions / max(total_actions, 1)) * 2
    scores["inside_forward"] = iff_score

    tf_score = float(max(0, avg_x_rel - 0.6) * 8)
    tf_score += shot_volume * 7
    tf_score += backward_pct * 2
    tf_score += (1.0 - wide_pct) * 4
    scores["target_forward"] = tf_score

    f9_score = max(0, avg_x_rel - 0.4) * 5
    f9_score += forward_pct * 5
    f9_score += (1.0 - shot_volume) * 5
    f9_score += (1.0 - wide_pct) * 3
    scores["false_nine"] = f9_score

    poacher_score = max(0, avg_x_rel - 0.7) * 8
    poacher_score += shot_volume * 8
    poacher_score += (1.0 - forward_pct) * 3
    poacher_score += (1.0 - wide_pct) * 5
    scores["poacher"] = poacher_score

    wp_score = wide_pct * 4
    wp_score += forward_pct * 5
    wp_score += (1.0 - shot_volume) * 5
    wp_score += len(crosses) / max(total_actions, 1) * 4
    scores["wide_playmaker"] = wp_score

    utility = (
        (wide_pct * 2.5)
        + ((1.0 - abs(avg_y_rel - 0.5) * 2) * 2.5)
        + (shot_volume * 2)
        + (def_actions / max(total_actions, 1) * 2)
    )
    scores["utility_player"] = utility

    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_roles[0][0] if sorted_roles else "unknown"
    secondary = sorted_roles[1][0] if len(sorted_roles) > 1 else ""
    top_score = sorted_roles[0][1] if sorted_roles else 0
    max_possible = 30.0
    confidence = min(top_score / max_possible, 1.0)

    return PlayerRole(
        primary_role=primary,
        secondary_role=secondary,
        confidence=round(confidence, 2),
        role_scores=dict(sorted_roles[:5]),
    )
