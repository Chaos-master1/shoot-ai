"""Trap → Transition Linkage — links pressing traps that lead to ball
recovery with subsequent counter-attacks and fast breaks."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from kawkab.core.pressing_traps import PressingTrap
from kawkab.core.transitions import PhaseTransition

TRAP_LINK_TIME_S = 3.0
TRAP_LINK_DISTANCE_M = 20.0
TRANSITION_LOOKAHEAD_S = 10.0


@dataclass
class TrapTransitionLink:
    trap_index: int
    transition_index: int
    time_delta: float
    spatial_distance: float
    goal_scored: bool
    shot_created: bool


@dataclass
class TrapTransitionAnalysis:
    total_traps: int
    successful_traps: int
    transitions_from_traps: list[TrapTransitionLink] = field(default_factory=list)
    conversion_rate: float = 0.0
    goal_conversion_rate: float = 0.0
    avg_transition_time: float = 0.0


def _zone_midpoint(trap: PressingTrap) -> tuple[float, float]:
    return (
        (trap.zone_x_range[0] + trap.zone_x_range[1]) / 2.0,
        (trap.zone_y_range[0] + trap.zone_y_range[1]) / 2.0,
    )


def _infer_team_for_trap(events: list[dict], trap: PressingTrap) -> str:
    from kawkab.core.pressing_traps import _classify_trap_zone

    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.get("type") not in ("tackle", "interception", "foul"):
            continue
        x = ev.get("x", ev.get("start_x"))
        y = ev.get("y", ev.get("start_y"))
        if x is None or y is None:
            continue
        if _classify_trap_zone(float(x), float(y)) == trap.zone_name:
            counts[ev.get("team", "")] += 1
    if not counts:
        return "home"
    return max(counts, key=counts.get)  # type: ignore[arg-type,return-value]


def _find_trap_recovery_events(
    events: list[dict],
    trap: PressingTrap,
    team: str,
) -> list[float]:
    from kawkab.core.pressing_traps import _classify_trap_zone

    sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0.0))
    n = len(sorted_ev)
    recovery_times: list[float] = []

    for i, ev in enumerate(sorted_ev):
        if ev.get("team") != team or ev.get("type") not in ("tackle", "interception", "foul"):
            continue
        x = ev.get("x", ev.get("start_x"))
        y = ev.get("y", ev.get("start_y"))
        if x is None or y is None:
            continue
        if _classify_trap_zone(float(x), float(y)) != trap.zone_name:
            continue
        for k in range(i + 1, min(n, i + 8)):
            later = sorted_ev[k]
            if later.get("team") == team and later.get("type") in (
                "pass",
                "carry",
                "shot",
                "dribble",
            ):
                recovery_times.append(later.get("timestamp", 0.0))
                break

    return recovery_times


def analyze_trap_transitions(
    traps: list[PressingTrap],
    transitions: list[PhaseTransition],
    events: list[dict],
) -> TrapTransitionAnalysis:
    if not traps:
        return TrapTransitionAnalysis(total_traps=0, successful_traps=0)

    successful_traps = [t for t in traps if t.regain_possession_count > 0]
    successful_count = len(successful_traps)
    links: list[TrapTransitionLink] = []
    total_time_delta = 0.0

    sorted_trans = sorted(transitions, key=lambda t: t.timestamp)

    for ti, trap in enumerate(successful_traps):
        team = _infer_team_for_trap(events, trap)
        trap_mid = _zone_midpoint(trap)
        recovery_times = _find_trap_recovery_events(events, trap, team)

        for recovery_ts in recovery_times:
            for tj, trans in enumerate(sorted_trans):
                if trans.team != team:
                    continue
                time_delta = trans.timestamp - recovery_ts
                if time_delta < 0 or time_delta > TRAP_LINK_TIME_S:
                    continue

                spatial_dist = math.sqrt(
                    (trans.start_x - trap_mid[0]) ** 2 + (trans.start_y - trap_mid[1]) ** 2
                )
                if spatial_dist > TRAP_LINK_DISTANCE_M:
                    continue

                shot_created = False
                goal_scored = False
                lookahead_limit = trans.timestamp + TRANSITION_LOOKAHEAD_S
                for ev in sorted(events, key=lambda e: e.get("timestamp", 0)):
                    ets = ev.get("timestamp", 0)
                    if ets <= trans.timestamp:
                        continue
                    if ets > lookahead_limit:
                        break
                    if ev.get("team") == team and ev.get("type") == "shot":
                        shot_created = True
                        if ev.get("is_goal"):
                            goal_scored = True

                links.append(
                    TrapTransitionLink(
                        trap_index=ti,
                        transition_index=tj,
                        time_delta=time_delta,
                        spatial_distance=spatial_dist,
                        goal_scored=goal_scored,
                        shot_created=shot_created,
                    )
                )
                total_time_delta += time_delta

    traps_with_links = {link.trap_index for link in links}
    traps_with_goals = {link.trap_index for link in links if link.goal_scored}
    conversion_rate = len(traps_with_links) / max(successful_count, 1)
    goal_conversion_rate = len(traps_with_goals) / max(successful_count, 1)
    avg_time = total_time_delta / max(len(links), 1)

    return TrapTransitionAnalysis(
        total_traps=len(traps),
        successful_traps=successful_count,
        transitions_from_traps=links,
        conversion_rate=conversion_rate,
        goal_conversion_rate=goal_conversion_rate,
        avg_transition_time=avg_time,
    )


def summarize_trap_transition(analysis: TrapTransitionAnalysis) -> dict[str, str]:
    if analysis.total_traps > 0:
        minutes_per = 90.0 / analysis.total_traps
        freq = f"A pressing trap every {minutes_per:.1f} minutes leading to a counter-attack"
    else:
        freq = "No pressing traps detected"

    if analysis.successful_traps > 0:
        chance_pct = analysis.conversion_rate * 100
        chance = f"{chance_pct:.0f}% of pressing traps result in scoring chances"
    else:
        chance = "0% of pressing traps result in scoring chances"

    if analysis.transitions_from_traps:
        avg = (
            "Average time from trap recovery to counter:"
            f" {analysis.avg_transition_time:.1f} seconds"
        )
    else:
        avg = "No trap-to-transition links found"

    return {
        "trap_frequency": freq,
        "chance_conversion": chance,
        "avg_transition_time": avg,
    }
