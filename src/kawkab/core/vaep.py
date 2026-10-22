"""VAEP (Value Added Event Possession) — possession-phase survival model.

Upgraded with possession-phase identification, Poisson goal arrivals,
and survival-based event valuation.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from kawkab.core.coordinate_validator import CoordinateValidator
from kawkab.core.game_constants import GAME
from kawkab.core.perf_timing import timed

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
X_ZONES = 16
Y_ZONES = 12
ZONE_WIDTH = PITCH_LENGTH / X_ZONES
ZONE_HEIGHT = PITCH_WIDTH / Y_ZONES
LOOKAHEAD_SECONDS = 10.0
SURVIVAL_DECAY = 0.85  # Discount factor per second for future events


def _to_zone(x: float, y: float) -> tuple[int, int]:
    zx = min(int(x / ZONE_WIDTH), X_ZONES - 1)
    zy = min(int(y / ZONE_HEIGHT), Y_ZONES - 1)
    return (zx, zy)


def _possession_switching_events() -> set[str]:
    return {
        "tackle",
        "interception",
        "clearance",
        "block",
        "ball_recovery",
        "dribble_past",
        "miscontrol",
        "foul",
        "own_goal",
    }


def _identify_possession_phases(
    events: list[dict[str, Any]],
) -> list[tuple[int, int, str]]:
    """Identify possession phases from event sequence.

    Returns:
        List of (start_idx, end_idx, team) for each possession phase.
    """
    phases: list[tuple[int, int, str]] = []
    if not events:
        return phases

    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0))
    n = len(sorted_ev)
    phase_start = 0
    current_team = sorted_ev[0].get("team", "home")
    switching = _possession_switching_events()

    for i in range(1, n):
        ev = sorted_ev[i]
        ev_type = ev.get("type", "")
        ev_team = ev.get("team", current_team)
        # Check for possession switch
        is_switch = False
        if (
            ev_type in switching
            or ev_type == "pass"
            and ev_team != current_team
            or ev_type == "shot"
            and ev_team != current_team
            or ev_team != current_team
            and ev_type in ("carry", "dribble", "receival")
        ):
            is_switch = True

        if is_switch:
            phases.append((phase_start, i - 1, current_team))
            phase_start = i
            current_team = ev_team

    # Last phase
    phases.append((phase_start, n - 1, current_team))
    return phases


def _estimate_poisson_rates(
    events: list[dict[str, Any]],
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """Estimate Poisson goal arrival rates per zone and team.

    Returns:
        (home_attack_rate, away_attack_rate, home_defend_rate, away_defend_rate)
        Each is a dict mapping zone -> lambda (goals per event).
    """
    home_goals: dict[tuple[int, int], int] = defaultdict(int)
    away_goals: dict[tuple[int, int], int] = defaultdict(int)
    home_att_events: dict[tuple[int, int], int] = defaultdict(int)
    away_att_events: dict[tuple[int, int], int] = defaultdict(int)
    home_def_events: dict[tuple[int, int], int] = defaultdict(int)
    away_def_events: dict[tuple[int, int], int] = defaultdict(int)

    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0))
    current_team = sorted_ev[0].get("team", "home") if sorted_ev else "home"

    for ev in sorted_ev:
        team = ev.get("team", current_team)
        zx, zy = _to_zone(ev.get("x", 52.5), ev.get("y", 34.0))
        zone = (zx, zy)
        ev_type = ev.get("type", "")

        if ev_type == "shot":
            if team == "home":
                home_att_events[zone] += 1
                if ev.get("is_goal"):
                    home_goals[zone] += 1
            else:
                away_att_events[zone] += 1
                if ev.get("is_goal"):
                    away_goals[zone] += 1
            # Opponent defends when shot is taken
            if team == "home":
                away_def_events[zone] += 1
            else:
                home_def_events[zone] += 1

        # Track defensive events
        if ev_type in _possession_switching_events():
            if team == "home":
                home_def_events[zone] += 1
            else:
                away_def_events[zone] += 1

        current_team = team

    # Laplace-smoothed rate estimation
    home_attack_rate: dict[tuple[int, int], float] = {}
    away_attack_rate: dict[tuple[int, int], float] = {}
    home_defend_rate: dict[tuple[int, int], float] = {}
    away_defend_rate: dict[tuple[int, int], float] = {}

    for zx in range(X_ZONES):
        for zy in range(Y_ZONES):
            zone = (zx, zy)
            ha = home_att_events.get(zone, 0)
            aa = away_att_events.get(zone, 0)
            hd = home_def_events.get(zone, 0)
            ad = away_def_events.get(zone, 0)

            home_attack_rate[zone] = (home_goals.get(zone, 0) + 0.01) / (ha + 0.1)
            away_attack_rate[zone] = (away_goals.get(zone, 0) + 0.01) / (aa + 0.1)
            home_defend_rate[zone] = (away_goals.get(zone, 0) + 0.01) / (ad + 0.1)
            away_defend_rate[zone] = (home_goals.get(zone, 0) + 0.01) / (hd + 0.1)

    return home_attack_rate, away_attack_rate, home_defend_rate, away_defend_rate


def _compute_player_relative_features(
    event: dict[str, Any],
    event_idx: int,
    all_events: list[dict[str, Any]],
    possession_event_indices: list[int],
) -> dict[str, Any]:
    ts = event.get("timestamp", 0.0)
    x = event.get("x", 52.5)
    y = event.get("y", 34.0)
    team = event.get("team", "home")
    ev_type = event.get("type", "")

    attack_dir = 1 if team == "home" else -1
    defensive_types = {"tackle", "interception", "clearance", "block", "ball_recovery"}

    num_defenders_ahead = 0
    num_teammates_nearby = 0
    min_defender_dist_behind = float("inf")

    for j in range(event_idx - 1, -1, -1):
        prev = all_events[j]
        prev_ts = prev.get("timestamp", 0.0)
        if ts - prev_ts > 5.0:
            break
        prev_team = prev.get("team", "")
        prev_type = prev.get("type", "")
        prev_x = prev.get("x", 52.5)
        prev_y = prev.get("y", 34.0)
        dist = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

        if (
            prev_team != team
            and prev_type in defensive_types
            and dist <= 20.0
            and attack_dir * (prev_x - x) >= 0
        ):
            num_defenders_ahead += 1

        if prev_team == team and dist <= 15.0:
            num_teammates_nearby += 1

        if prev_team != team and prev_type in defensive_types and attack_dir * (prev_x - x) <= 0:
            behind_dist = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            min_defender_dist_behind = min(min_defender_dist_behind, behind_dist)

    speed_of_attack = 0.0
    possession_times = []
    for idx in possession_event_indices:
        if idx <= event_idx:
            possession_times.append(all_events[idx].get("timestamp", 0.0))
    if len(possession_times) >= 3:
        gaps = []
        for k in range(1, len(possession_times)):
            gaps.append(possession_times[k] - possession_times[k - 1])
        if gaps:
            n = min(3, len(gaps))
            speed_of_attack = sum(gaps[-n:]) / n

    is_through_ball = False
    if ev_type == "pass" and num_defenders_ahead > 0:
        end_x = event.get("end_x", x)
        if attack_dir * (end_x - x) > 0:
            is_through_ball = True

    space_behind = min_defender_dist_behind == float("inf") or min_defender_dist_behind > 10.0

    return {
        "num_defenders_ahead": num_defenders_ahead,
        "num_teammates_nearby": num_teammates_nearby,
        "speed_of_attack": round(speed_of_attack, 2),
        "is_through_ball": is_through_ball,
        "space_behind": space_behind,
    }


@dataclass
class VaepepResult:
    event_index: int = 0
    event_type: str = ""
    timestamp: float = 0.0
    team: str = "home"
    zone_x: int = 0
    zone_y: int = 0
    delta_home: float = 0.0
    delta_away: float = 0.0
    vaep_value: float = 0.0
    is_goal: bool = False
    possession_id: int = -1
    survival_pre: float = 0.0
    survival_post: float = 0.0
    num_defenders_ahead: int = 0
    num_teammates_nearby: int = 0
    speed_of_attack: float = 0.0
    is_through_ball: bool = False
    space_behind: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "event_type": self.event_type,
            "timestamp": round(self.timestamp, 1),
            "team": self.team,
            "zone": f"{self.zone_x}_{self.zone_y}",
            "delta_home": round(self.delta_home, 4),
            "delta_away": round(self.delta_away, 4),
            "vaep_value": round(self.vaep_value, 4),
            "is_goal": self.is_goal,
            "possession_id": self.possession_id,
            "num_defenders_ahead": self.num_defenders_ahead,
            "num_teammates_nearby": self.num_teammates_nearby,
            "speed_of_attack": round(self.speed_of_attack, 2),
            "is_through_ball": self.is_through_ball,
            "space_behind": self.space_behind,
        }


@timed()
def compute_vaep(
    events: list[dict[str, Any]],
    frames: list[dict[str, Any]] | None = None,
    lookahead: float = LOOKAHEAD_SECONDS,
    attacking_direction: str = "right",
) -> list[dict[str, Any]]:
    """Compute VAEP using possession-phase survival model.

    Args:
        events: Sorted list of event dicts (type, timestamp, team, x, y, is_goal).
        frames: Optional — not used in survival VAEP (kept for API compat).
        lookahead: Seconds to look ahead for goal probability.

    Returns:
        List of VaepepResult dicts sorted by timestamp.
    """
    for ev in events:
        CoordinateValidator.validate_event_spatial(ev)
    if not events:
        return []

    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0))
    _ = len(sorted_ev)

    # 1. Identify possession phases
    phases = _identify_possession_phases(sorted_ev)

    # 2. Estimate Poisson rates
    home_attack_rate, away_attack_rate, home_defend_rate, away_defend_rate = (
        _estimate_poisson_rates(sorted_ev)
    )

    def _survival_prob(zone: tuple[int, int], team: str, dt: float) -> tuple[float, float]:
        """Survival probability: P(no goal conceded) over dt seconds."""
        if team == "home":
            atk_rate = home_attack_rate.get(zone, 0.001)
            def_rate = away_defend_rate.get(zone, 0.001)
        else:
            atk_rate = away_attack_rate.get(zone, 0.001)
            def_rate = home_defend_rate.get(zone, 0.001)

        # Poisson process: P(0 goals) = exp(-λ * dt), decay applied per-second
        score_prob = 1.0 - math.exp(-atk_rate * dt * SURVIVAL_DECAY**dt)
        concede_prob = 1.0 - math.exp(-def_rate * dt * SURVIVAL_DECAY**dt)
        return score_prob, concede_prob

    # 3. Compute VAEP for each event
    phase_map: dict[int, int] = {}  # event_idx -> possession_id
    possession_events_map: dict[int, list[int]] = {}
    for pid, (start, end, _team) in enumerate(phases):
        idxs = list(range(start, end + 1))
        possession_events_map[pid] = idxs
        for idx in idxs:
            phase_map[idx] = pid

    results: list[VaepepResult] = []

    for i, ev in enumerate(sorted_ev):
        ts = ev.get("timestamp", 0.0)
        ev_x = ev.get("x", 52.5)
        if attacking_direction == "left":
            ev_x = PITCH_LENGTH - ev_x
        zx, zy = _to_zone(ev_x, ev.get("y", 34.0))
        team = ev.get("team", "home")
        zone = (zx, zy)
        pid = phase_map.get(i, -1)

        features = _compute_player_relative_features(
            ev, i, sorted_ev, possession_events_map.get(pid, [])
        )

        adj_score = (
            1.0
            + 0.05 * features["num_teammates_nearby"]
            + 0.2 * int(features["is_through_ball"])
            + 0.15 * int(features["space_behind"])
        )
        adj_concede = 1.0 + 0.1 * features["num_defenders_ahead"]

        if team == "home":
            atk_rate = home_attack_rate.get(zone, 0.001) * adj_score
            def_rate = away_defend_rate.get(zone, 0.001) * adj_concede
        else:
            atk_rate = away_attack_rate.get(zone, 0.001) * adj_score
            def_rate = home_defend_rate.get(zone, 0.001) * adj_concede

        def _prob(rate: float, dt: float) -> float:
            return 1.0 - math.exp(-rate * dt * SURVIVAL_DECAY**dt)

        pre_score_prob = _prob(atk_rate, lookahead)
        pre_concede_prob = _prob(def_rate, lookahead)
        pre_value = pre_score_prob - pre_concede_prob

        # Compute post-event probabilities based on new game state
        is_shot = ev.get("type") in ("shot", "goal")
        is_turnover = ev.get("type") in _possession_switching_events()
        is_goal = ev.get("is_goal", False)

        if is_goal:
            post_score_prob = 1.0
            post_concede_prob = 0.0
        elif is_turnover:
            post_score_prob = 0.0
            # Opponent now has possession from this zone
            post_concede_prob = _prob(def_rate, lookahead * 0.5)
        elif is_shot:
            # Shot taken - post-event probability drops (ball no longer in dangerous area)
            post_score_prob = _prob(atk_rate, lookahead * 0.3)
            post_concede_prob = _prob(def_rate, lookahead * 0.3)
        else:
            # Pass/dribble/carry: move to next event's position if same possession
            next_pos = None
            for j in range(i + 1, min(i + 5, len(sorted_ev))):
                nxt = sorted_ev[j]
                if nxt.get("team") == team and phase_map.get(j, -1) == pid:
                    next_pos = (nxt.get("x", 52.5), nxt.get("y", 34.0))
                    break
            if next_pos:
                nxt_x = next_pos[0]
                if attacking_direction == "left":
                    nxt_x = PITCH_LENGTH - nxt_x
                nzx, nzy = _to_zone(nxt_x, next_pos[1])
                nzone = (nzx, nzy)
                if team == "home":
                    next_atk = home_attack_rate.get(nzone, 0.001) * adj_score
                    next_def = away_defend_rate.get(nzone, 0.001) * adj_concede
                else:
                    next_atk = away_attack_rate.get(nzone, 0.001) * adj_score
                    next_def = home_defend_rate.get(nzone, 0.001) * adj_concede
                post_score_prob = _prob(next_atk, lookahead)
                post_concede_prob = _prob(next_def, lookahead)
            else:
                # Possession ended or no next event - reduce probability
                post_score_prob = _prob(atk_rate, lookahead * 0.5)
                post_concede_prob = _prob(def_rate, lookahead * 0.5)

        post_value = post_score_prob - post_concede_prob

        delta_home = post_score_prob - pre_score_prob
        delta_away = post_concede_prob - pre_concede_prob
        vaep = delta_home - delta_away

        results.append(
            VaepepResult(
                event_index=i,
                event_type=ev.get("type", "unknown"),
                timestamp=ts,
                team=team,
                zone_x=zx,
                zone_y=zy,
                delta_home=delta_home,
                delta_away=delta_away,
                vaep_value=vaep,
                is_goal=ev.get("is_goal", False),
                possession_id=pid,
                survival_pre=pre_value,
                survival_post=post_value,
                num_defenders_ahead=features["num_defenders_ahead"],
                num_teammates_nearby=features["num_teammates_nearby"],
                speed_of_attack=features["speed_of_attack"],
                is_through_ball=features["is_through_ball"],
                space_behind=features["space_behind"],
            )
        )

    return [r.to_dict() for r in results]


def compute_vaep_with_ci(
    events: list[dict[str, Any]],
    frames: list[dict[str, Any]] | None = None,
    lookahead: float = LOOKAHEAD_SECONDS,
    n_bootstrap: int = 100,
) -> list[dict[str, Any]]:
    """Compute VAEP with block-bootstrap confidence intervals.

    Resamples possession phases (blocks) with replacement to preserve
    temporal dependence, then computes per-event CIs from the bootstrap
    distribution.

    Args:
        events: Sorted list of event dicts.
        frames: Optional — passed through to compute_vaep.
        lookahead: Lookahead window in seconds.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        List of VAEP result dicts with added ci_lower and ci_upper keys.
    """
    if not events:
        return []

    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0))
    phases = _identify_possession_phases(sorted_ev)
    phase_ids = list(range(len(phases)))

    point_results = compute_vaep(sorted_ev, frames, lookahead)

    per_event_values: dict[int, list[float]] = {}
    for r in point_results:
        per_event_values[r["event_index"]] = [r["vaep_value"]]

    for _ in range(n_bootstrap):
        sampled_ids = random.choices(phase_ids, k=len(phase_ids))
        temp_pairs: list[tuple[float, int, dict[str, Any]]] = []
        for pid in sampled_ids:
            start, end, _ = phases[pid]
            for idx in range(start, end + 1):
                ev = sorted_ev[idx]
                temp_pairs.append((ev.get("timestamp", 0.0), idx, ev))

        temp_pairs.sort(key=lambda x: x[0])
        sorted_indices = [p[1] for p in temp_pairs]
        sampled_events_list = [p[2] for p in temp_pairs]

        resampled = compute_vaep(sampled_events_list, frames, lookahead)
        for r in resampled:
            idx = r["event_index"]
            if idx < len(sorted_indices):
                orig_idx = sorted_indices[idx]
                if orig_idx in per_event_values:
                    per_event_values[orig_idx].append(r["vaep_value"])

    final_results: list[dict[str, Any]] = []
    for r in point_results:
        idx = r["event_index"]
        vals = per_event_values.get(idx, [r["vaep_value"]])
        if len(vals) < 2:
            ci_lower = r["vaep_value"]
            ci_upper = r["vaep_value"]
        else:
            ci_lower = float(np.percentile(vals, 2.5))
            ci_upper = float(np.percentile(vals, 97.5))
        final_results.append(
            {
                **r,
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "n_bootstrap": n_bootstrap,
            }
        )

    return final_results


def compute_vaep_v2(
    events: list[dict[str, Any]],
    frames: list[dict[str, Any]] | None = None,
    lookahead: float = LOOKAHEAD_SECONDS,
) -> list[dict[str, Any]]:
    """Compute VAEP 2.0 using frame-level tracking data.

    Unlike v1 which uses possession-phase survival probability,
    v2 computes per-frame pitch control and values events by
    the change in home/away control probability before and after.

    Args:
        events: Sorted list of event dicts with timestamps.
        frames: List of tracking frame dicts, each with:
            - timestamp: float
            - home_positions: list of [x, y] per home player
            - away_positions: list of [x, y] per away player
            - ball_x, ball_y: float
        lookahead: Seconds to look ahead for scoring probability.

    Returns:
        List of enhanced VaepepResult dicts with frame-based valuation.
    """
    if not frames:
        return compute_vaep(events, None, lookahead)

    from kawkab.core.pitch_control import WeightedPitchControl
    from kawkab.core.xg_model import compute_xg_trained_from_dict

    results = []
    pc_model = WeightedPitchControl()

    for _i, ev in enumerate(events):
        ts = ev.get("timestamp", 0.0)
        team = ev.get("team", "home")

        frame = _find_closest_frame(frames, ts)
        if frame is None:
            results.append(
                {
                    "timestamp": ts,
                    "team": team,
                    "type": ev.get("type", "unknown"),
                    "vaep": 0.0,
                    "method": "fallback",
                }
            )
            continue

        home_pos = frame.get("home_positions", [])
        away_pos = frame.get("away_positions", [])
        ball_x = frame.get("ball_x", 52.5)
        ball_y = frame.get("ball_y", 34.0)

        pre_control = pc_model.compute_frame_control(
            home_positions=home_pos,
            away_positions=away_pos,
            ball_pos=(ball_x, ball_y),
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
        )

        next_frame = _find_closest_frame(frames, ts + 2.0)
        if next_frame:
            post_control = pc_model.compute_frame_control(
                home_positions=next_frame.get("home_positions", home_pos),
                away_positions=next_frame.get("away_positions", away_pos),
                ball_pos=(next_frame.get("ball_x", ball_x), next_frame.get("ball_y", ball_y)),
                pitch_length=PITCH_LENGTH,
                pitch_width=PITCH_WIDTH,
            )

            delta_home = post_control.home_control_pct - pre_control.home_control_pct
            delta_away = post_control.away_control_pct - pre_control.away_control_pct

            dist_to_goal = math.sqrt((PITCH_LENGTH - ball_x) ** 2 + (PITCH_WIDTH / 2 - ball_y) ** 2)
            angle_to_goal = math.degrees(
                math.atan2(PITCH_WIDTH / 2 - ball_y, PITCH_LENGTH - ball_x)
            )
            # Trained xG as the scoring-probability term (the legacy
            # compute_xg heuristic underestimated xG ~5x, deflating every
            # frame-based VAEP value with it).
            scoring_prob = compute_xg_trained_from_dict(
                {
                    "type": "shot",
                    "distance_m": dist_to_goal,
                    "angle_deg": abs(angle_to_goal),
                }
            )

            vaep = (delta_home - delta_away) * scoring_prob
        else:
            vaep = 0.0

        results.append(
            {
                "timestamp": ts,
                "team": team,
                "type": ev.get("type", "unknown"),
                "vaep": round(vaep, 4),
                "method": "frame_based",
                "pre_home_control": round(pre_control.home_control_pct, 1),
                "pre_away_control": round(pre_control.away_control_pct, 1),
            }
        )

    return results


def _find_closest_frame(
    frames: list[dict[str, Any]],
    target_ts: float,
) -> dict[str, Any] | None:
    """Find the frame closest to the given timestamp."""
    if not frames:
        return None
    best = None
    best_dist = float("inf")
    for f in frames:
        dist = abs(f.get("timestamp", 0.0) - target_ts)
        if dist < best_dist:
            best_dist = dist
            best = f
    return best
