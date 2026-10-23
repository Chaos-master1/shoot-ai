"""Style profile — the same six measurements on both sides of a comparison.

Kawkab's benchmark started with outputs (shots, xG, goals). Elite tools
benchmark *style* — how a team pressed, where it played, how it
progressed the ball — because style is what a coach can change before
the next match. This module owns those measurements and nothing else.

Every input is the same normalized event stream on both sides of the
comparison: a list of dicts with ``type``, ``x``/``y``,
``end_x``/``end_y``, ``completed`` and ``won``. Coordinates are Kawkab
meters (105 x 68), attacking toward x = 105, and the stream is
possession-relative: each team's events are measured toward the goal it
attacks. Both the corpus loader and the stored-event adapter produce
this shape, which is what makes the percentile a like-for-like
comparison.

Thresholds are chosen once, here, and recorded in the artifact through
the metric contract — a metric whose definition drifts between builds
is worse than no metric.

Pure functions. No I/O, no service imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kawkab.core.elite_reference import per_90

# ── Pitch geometry (Kawkab meters) ────────────────────────────────────────
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_CENTER_Y = 34.0

#: Final third begins here (105 - 35).
FINAL_THIRD_X = 70.0
#: The pressing zone: the opponent's own 60% of the pitch.
PRESS_ZONE_X = 63.0
#: A pass is progressive when it advances at least this much toward goal.
PROGRESSIVE_DISTANCE_M = 10.0
#: ...from the halfway line onward (a 10 m pass inside a low block is not
#: progression in the sense every provider means).
HALFWAY_X = PITCH_LENGTH_M / 2.0
#: Penalty area: 16.5 m deep, 40.32 m wide.
BOX_X = 88.5
BOX_HALF_WIDTH_M = 20.16
#: A turnover is "high" when won within 40 m of the opponent's goal.
HIGH_TURNOVER_X = 65.0

# ── Coverage minimums ────────────────────────────────────────────────────
# A metric built from 8 passes is noise wearing a number's clothes. Each
# metric states the input it needs; below it the service reports the
# count and the reason instead of a value.
MIN_DEFENSIVE_ACTIONS = 5
MIN_OPPONENT_BUILD_PASSES = 20
MIN_FINAL_THIRD_PASSES = 20
MIN_LOCATED_PASSES = 50
MIN_LOCATED_ENTRIES = 80
MIN_LOCATED_DEFENSIVE = 10

#: PPDA's denominator. Pressures are deliberately excluded: event feeds
#: record hundreds of pressure events per match, which drives PPDA to ~2
#: regardless of how the team actually pressed. Measured on this cohort,
#: tackles + interceptions + fouls reproduce the published 4-16 band.
DEFENSIVE_ACTIONS = ("interception", "duel", "foul")
TURNOVER_ACTIONS = ("interception", "recovery", "duel")


def _located(event: Mapping[str, Any]) -> bool:
    """True when the event carries a usable position on the pitch."""
    x = event.get("x")
    y = event.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    return -0.5 <= x <= PITCH_LENGTH_M + 0.5 and -0.5 <= y <= PITCH_WIDTH_M + 0.5


def _in_box(x: float, y: float) -> bool:
    return x >= BOX_X and abs(y - GOAL_CENTER_Y) <= BOX_HALF_WIDTH_M


def _coverage(n: int, minimum: int, unit: str) -> dict[str, Any]:
    return {"n": int(n), "minimum": int(minimum), "unit": unit}


def _unavailable(unit: str, n: int, minimum: int) -> tuple[None, dict[str, Any]]:
    return None, _coverage(n, minimum, unit)


def _ppda(own: Sequence[Mapping[str, Any]], opponent: Sequence[Mapping[str, Any]]):
    their_build = [
        e for e in opponent
        if e.get("type") == "pass" and _located(e) and e["x"] < PRESS_ZONE_X
    ]
    our_press = [
        e for e in own
        if e.get("type") in DEFENSIVE_ACTIONS and _located(e) and e["x"] < PRESS_ZONE_X
    ]
    if len(our_press) < MIN_DEFENSIVE_ACTIONS:
        return _unavailable(
            "defensive actions in the pressing zone",
            len(our_press),
            MIN_DEFENSIVE_ACTIONS,
        )
    if len(their_build) < MIN_OPPONENT_BUILD_PASSES:
        return _unavailable(
            "opponent build-up passes",
            len(their_build),
            MIN_OPPONENT_BUILD_PASSES,
        )
    return len(their_build) / len(our_press), _coverage(
        len(our_press), MIN_DEFENSIVE_ACTIONS, "defensive actions in the pressing zone"
    )


def _field_tilt(own: Sequence[Mapping[str, Any]], opponent: Sequence[Mapping[str, Any]]):
    def final_third_completions(events: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            1
            for e in events
            if e.get("type") == "pass"
            and e.get("completed") is True
            and _located(e)
            and e["x"] >= FINAL_THIRD_X
        )

    ours = final_third_completions(own)
    theirs = final_third_completions(opponent)
    total = ours + theirs
    if total < MIN_FINAL_THIRD_PASSES:
        return _unavailable(
            "completed final-third passes (both teams)", total, MIN_FINAL_THIRD_PASSES
        )
    return ours / total, _coverage(
        total, MIN_FINAL_THIRD_PASSES, "completed final-third passes (both teams)"
    )


def _pass_outcomes_known(passes: Sequence[Mapping[str, Any]]) -> bool:
    return all(e.get("completed") is not None for e in passes)


def _progressive_passes(own: Sequence[Mapping[str, Any]]):
    passes = [e for e in own if e.get("type") == "pass" and _located(e)]
    if len(passes) < MIN_LOCATED_PASSES:
        return _unavailable("located passes", len(passes), MIN_LOCATED_PASSES)
    if not _pass_outcomes_known(passes):
        known = sum(1 for e in passes if e.get("completed") is not None)
        return _unavailable("passes with a recorded outcome", known, len(passes))
    count = 0
    for e in passes:
        end_x = e.get("end_x")
        end_y = e.get("end_y")
        if e.get("completed") is not True:
            continue
        if not isinstance(end_x, (int, float)) or not isinstance(end_y, (int, float)):
            continue
        if _in_box(end_x, end_y) or (
            e["x"] >= HALFWAY_X and end_x - e["x"] >= PROGRESSIVE_DISTANCE_M
        ):
            count += 1
    return count, _coverage(len(passes), MIN_LOCATED_PASSES, "located passes")


def _final_third_entries(own: Sequence[Mapping[str, Any]]):
    candidates = [
        e for e in own
        if e.get("type") in ("pass", "carry") and _located(e)
    ]
    if len(candidates) < MIN_LOCATED_ENTRIES:
        return _unavailable(
            "located passes and carries", len(candidates), MIN_LOCATED_ENTRIES
        )
    count = 0
    for e in candidates:
        end_x = e.get("end_x")
        if not isinstance(end_x, (int, float)):
            continue
        if e.get("type") == "pass" and e.get("completed") is not True:
            continue
        if e["x"] < FINAL_THIRD_X <= end_x:
            count += 1
    return count, _coverage(
        len(candidates), MIN_LOCATED_ENTRIES, "located passes and carries"
    )


def _high_turnovers(own: Sequence[Mapping[str, Any]]):
    actions = [
        e for e in own
        if e.get("type") in TURNOVER_ACTIONS and _located(e)
    ]
    if len(actions) < MIN_LOCATED_DEFENSIVE:
        return _unavailable(
            "located defensive actions", len(actions), MIN_LOCATED_DEFENSIVE
        )
    count = sum(
        1
        for e in actions
        if e["x"] >= HIGH_TURNOVER_X
        and (e.get("type") != "duel" or e.get("won") is True)
    )
    return count, _coverage(
        len(actions), MIN_LOCATED_DEFENSIVE, "located defensive actions"
    )


def _pass_completion(own: Sequence[Mapping[str, Any]]):
    passes = [e for e in own if e.get("type") == "pass" and _located(e)]
    known = [e for e in passes if e.get("completed") is not None]
    if len(known) < MIN_LOCATED_PASSES:
        return _unavailable(
            "passes with a recorded outcome", len(known), MIN_LOCATED_PASSES
        )
    if len(known) != len(passes):
        # A partial completion rate would understate or overstate the
        # team; report the missing outcomes instead of guessing.
        return _unavailable(
            "passes with a recorded outcome", len(known), len(passes)
        )
    completed = sum(1 for e in known if e["completed"] is True)
    return 100.0 * completed / len(known), _coverage(
        len(known), MIN_LOCATED_PASSES, "passes with a recorded outcome"
    )


def build_profile(
    own: Sequence[Mapping[str, Any]],
    opponent: Sequence[Mapping[str, Any]],
    minutes: float,
) -> dict[str, Any]:
    """The six style measurements for one team in one match.

    ``own`` and ``opponent`` are the same normalized stream for each
    side. ``minutes`` scales the per-90 metrics; a match with an unknown
    length is normalised to 90 rather than 1 (which would multiply every
    rate by 90).
    """
    minutes = float(minutes) if minutes and minutes > 0 else 90.0
    minutes = max(1.0, min(minutes, 200.0))

    ppda, ppda_cov = _ppda(own, opponent)
    tilt, tilt_cov = _field_tilt(own, opponent)
    prog, prog_cov = _progressive_passes(own)
    entries, entries_cov = _final_third_entries(own)
    turnovers, turnovers_cov = _high_turnovers(own)
    completion, completion_cov = _pass_completion(own)

    metrics = {
        "ppda": ppda,
        "field_tilt": tilt,
        "progressive_passes_per_90": None if prog is None else per_90(prog, minutes),
        "final_third_entries_per_90": None if entries is None else per_90(entries, minutes),
        "high_turnovers_per_90": None if turnovers is None else per_90(turnovers, minutes),
        "pass_completion_pct": completion,
    }
    return {
        "metrics": metrics,
        "coverage": {
            "ppda": ppda_cov,
            "field_tilt": tilt_cov,
            "progressive_passes_per_90": prog_cov,
            "final_third_entries_per_90": entries_cov,
            "high_turnovers_per_90": turnovers_cov,
            "pass_completion_pct": completion_cov,
        },
        "minutes": minutes,
    }


def unavailable_reason(key: str, profile: Mapping[str, Any]) -> str:
    """Why a style metric could not be measured, in the counts observed."""
    coverage = (profile.get("coverage") or {}).get(key) or {}
    n = coverage.get("n")
    minimum = coverage.get("minimum")
    unit = coverage.get("unit") or "events"
    if n is None or minimum is None:
        return "the events for this match do not carry the positions this metric needs"
    return f"needs at least {minimum} {unit}, found {n}"
