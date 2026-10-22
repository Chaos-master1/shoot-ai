"""Detect anomalies in match event data."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnomalyReport:
    anomalies: list[dict] = field(default_factory=list)
    score: float = 100.0


MAX_PLAYER_SPEED_MS = 10.0
MAX_GOALS = 15
MAX_PERIOD_GAP_S = 300.0
MAX_SUBS_PER_TEAM = 5
GOAL_TYPE = "goal"
SHOT_TYPE = "shot"


def _compute_score_internal(
    events: list[dict], match_duration_min: float, anomalies: list[dict]
) -> float:
    if not events:
        return 0.0

    score = 100.0
    severity_penalties = {"high": 20.0, "medium": 10.0, "low": 3.0}
    for anomaly in anomalies:
        sev = anomaly.get("severity", "low")
        score -= severity_penalties.get(sev, 3.0)

    if len(events) < 10:
        score -= 30.0
    n_timestamps = sum(1 for e in events if isinstance(e.get("timestamp"), (int, float)))
    if n_timestamps < len(events) * 0.5:
        score -= 15.0
    n_shots = sum(1 for e in events if e.get("type") == "shot")
    if n_shots == 0:
        score -= 10.0

    return max(0.0, min(100.0, score))


def detect_anomalies(events: list[dict], match_duration_min: float = 90.0) -> AnomalyReport:
    anomalies: list[dict] = []

    if not events:
        return AnomalyReport(
            anomalies=[
                {
                    "type": "no_events",
                    "severity": "high",
                    "description": "No events in match data",
                    "event_id": None,
                }
            ],
            score=0.0,
        )

    speed_anomalies = _detect_impossible_speed(events)
    anomalies.extend(speed_anomalies)

    goal_anomalies = _detect_too_many_goals(events)
    anomalies.extend(goal_anomalies)

    missing_goal_anomalies = _detect_missing_goals(events)
    anomalies.extend(missing_goal_anomalies)

    gap_anomalies = _detect_period_gaps(events, match_duration_min)
    anomalies.extend(gap_anomalies)

    dup_anomalies = _detect_duplicate_events(events)
    anomalies.extend(dup_anomalies)

    coord_anomalies = _detect_coordinate_outliers(events)
    anomalies.extend(coord_anomalies)

    sub_anomalies = _detect_too_many_subs(events)
    anomalies.extend(sub_anomalies)

    score = _compute_score_internal(events, match_duration_min, anomalies)

    return AnomalyReport(anomalies=anomalies, score=score)


def _detect_impossible_speed(events: list[dict]) -> list[dict]:
    anomalies: list[dict] = []
    for ev in events:
        speed = ev.get("speed_mps", 0.0)
        if speed is not None and isinstance(speed, (int, float)) and speed > MAX_PLAYER_SPEED_MS:
            anomalies.append(
                {
                    "type": "impossible_speed",
                    "severity": "high",
                    "description": f"Player speed {speed:.1f} m/s exceeds maximum {MAX_PLAYER_SPEED_MS} m/s",
                    "event_id": ev.get("id"),
                }
            )
    return anomalies


def _detect_too_many_goals(events: list[dict]) -> list[dict]:
    n_goals = sum(1 for e in events if e.get("type") == GOAL_TYPE or e.get("is_goal"))
    if n_goals > MAX_GOALS:
        return [
            {
                "type": "too_many_goals",
                "severity": "medium",
                "description": f"Match has {n_goals} goals, exceeding {MAX_GOALS}",
                "event_id": None,
            }
        ]
    return []


def _detect_missing_goals(events: list[dict]) -> list[dict]:
    anomalies: list[dict] = []
    for e in events:
        if e.get("type") != SHOT_TYPE:
            continue
        xg = e.get("xg", e.get("xG", 0.0))
        is_goal = e.get("is_goal", False)
        if isinstance(xg, (int, float)) and xg > 0.9 and not is_goal:
            anomalies.append(
                {
                    "type": "missing_goal_high_xg",
                    "severity": "low",
                    "description": f"High-xG shot (xg={xg:.3f}) did not result in goal",
                    "event_id": e.get("id"),
                }
            )
    return anomalies


def _detect_period_gaps(events: list[dict], match_duration_min: float) -> list[dict]:
    anomalies: list[dict] = []
    timestamps = sorted(
        [e.get("timestamp", 0.0) for e in events if isinstance(e.get("timestamp"), (int, float))]
    )
    if len(timestamps) < 2:
        return anomalies

    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > MAX_PERIOD_GAP_S:
            anomalies.append(
                {
                    "type": "period_gap",
                    "severity": "medium",
                    "description": f"Gap of {gap:.0f}s between events exceeds {MAX_PERIOD_GAP_S:.0f}s limit",
                    "event_id": None,
                }
            )

    total_duration = timestamps[-1] - timestamps[0]
    expected = match_duration_min * 60.0
    if total_duration < expected * 0.5:
        anomalies.append(
            {
                "type": "short_duration",
                "severity": "medium",
                "description": f"Match duration {total_duration:.0f}s is less than half expected ({expected:.0f}s)",
                "event_id": None,
            }
        )

    return anomalies


def _detect_duplicate_events(events: list[dict]) -> list[dict]:
    anomalies: list[dict] = []
    seen: set[tuple] = set()
    for e in events:
        ts = e.get("timestamp")
        et = e.get("event_type", e.get("type"))
        team = e.get("team")
        key = (ts, et, team)
        if key in seen:
            anomalies.append(
                {
                    "type": "duplicate_event",
                    "severity": "low",
                    "description": f"Duplicate event: {et} at {ts}s for {team}",
                    "event_id": e.get("id"),
                }
            )
        else:
            seen.add(key)
    return anomalies


def _detect_coordinate_outliers(events: list[dict]) -> list[dict]:
    anomalies: list[dict] = []
    for e in events:
        for coord_key in ("x", "y", "start_x", "start_y", "end_x", "end_y"):
            val = e.get(coord_key)
            if val is None or not isinstance(val, (int, float)):
                continue
            if coord_key in ("x", "start_x", "end_x") and (val < 0 or val > 105):
                anomalies.append(
                    {
                        "type": "coordinate_outlier",
                        "severity": "high",
                        "description": f"x-coordinate {val:.1f} outside pitch bounds [0, 105]",
                        "event_id": e.get("id"),
                    }
                )
            elif coord_key in ("y", "start_y", "end_y") and (val < 0 or val > 68):
                anomalies.append(
                    {
                        "type": "coordinate_outlier",
                        "severity": "high",
                        "description": f"y-coordinate {val:.1f} outside pitch bounds [0, 68]",
                        "event_id": e.get("id"),
                    }
                )
    return anomalies


def _detect_too_many_subs(events: list[dict]) -> list[dict]:
    anomalies: list[dict] = []
    team_subs: dict[str, int] = {}
    for e in events:
        if e.get("event_type") == "substitution" or e.get("type") == "substitution":
            team = e.get("team", "unknown")
            team_subs[team] = team_subs.get(team, 0) + 1

    for team, count in team_subs.items():
        if count > MAX_SUBS_PER_TEAM:
            anomalies.append(
                {
                    "type": "too_many_subs",
                    "severity": "medium",
                    "description": f"Team '{team}' has {count} substitutions, exceeding {MAX_SUBS_PER_TEAM}",
                    "event_id": None,
                }
            )
    return anomalies


def compute_data_quality_score(events: list[dict], match_duration_min: float = 90.0) -> float:
    if not events:
        return 0.0
    anomalies = detect_anomalies(events, match_duration_min)
    return anomalies.score
