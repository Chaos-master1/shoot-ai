"""Per-player style measurements — the player-level sibling of style_profile.

A player benchmark needs the same kinds of measurements as the team one, but
a player touches the ball far less than a team does, so the team's coverage
floors (50 located passes) would reject almost every player. These are the
player floors, chosen once and stated in every refusal:

- 30 minutes: a per-90 rate over a 12-minute cameo is noise;
- 10 located passes for passing metrics;
- 10 located passes/carries for final-third entries;
- 3 located defensive actions for high turnovers.

The predicates (progressive pass, final-third entry, high turnover) come
from ``core.style_profile`` — the same definitions the team benchmark uses.
Output metrics (shots, xG, goals) are filled by the caller, which owns the
shot model and the goal-outcome decision.

Pure functions. No I/O, no service imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kawkab.core.elite_reference import per_90
from kawkab.core.style_profile import (
    is_final_third_entry,
    is_high_turnover,
    is_located,
    is_progressive_pass,
)

#: A player needs this many minutes for a per-90 percentile.
MIN_PLAYER_MINUTES = 30.0
#: Passing metrics need at least this many located passes.
MIN_PLAYER_PASSES = 10
#: Entry metrics need at least this many located passes/carries.
MIN_PLAYER_ENTRIES = 10
#: High-turnover metrics need at least this many located defensive actions.
MIN_PLAYER_DEF_ACTIONS = 3


def _coverage(n: int, minimum: int, unit: str) -> dict[str, Any]:
    return {"n": int(n), "minimum": int(minimum), "unit": unit}


def _pass_metrics(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    passes = [e for e in events if e.get("type") == "pass" and is_located(e)]
    known = [e for e in passes if e.get("completed") is not None]
    metrics: dict[str, Any] = {
        "passes_per_90": None,
        "progressive_passes_per_90": None,
        "pass_completion_pct": None,
    }
    coverage: dict[str, Any] = {}

    if len(passes) < MIN_PLAYER_PASSES:
        for key in metrics:
            coverage[key] = _coverage(len(passes), MIN_PLAYER_PASSES, "located passes")
        return metrics, coverage

    coverage["passes_per_90"] = _coverage(len(passes), MIN_PLAYER_PASSES, "located passes")
    # A pass whose outcome is missing cannot be judged progressive or
    # complete; report the missing outcomes instead of silently treating
    # them as failures.
    if len(known) != len(passes):
        coverage["progressive_passes_per_90"] = _coverage(
            len(known), len(passes), "passes with a recorded outcome"
        )
        coverage["pass_completion_pct"] = _coverage(
            len(known), len(passes), "passes with a recorded outcome"
        )
        return metrics, coverage

    coverage["progressive_passes_per_90"] = _coverage(
        len(passes), MIN_PLAYER_PASSES, "located passes"
    )
    coverage["pass_completion_pct"] = _coverage(
        len(passes), MIN_PLAYER_PASSES, "passes with a recorded outcome"
    )
    progressive = sum(1 for e in passes if is_progressive_pass(e))
    completed = sum(1 for e in passes if e.get("completed") is True)
    metrics["progressive_passes_per_90"] = progressive
    metrics["pass_completion_pct"] = 100.0 * completed / len(passes)
    metrics["passes_per_90"] = len(passes)
    return metrics, coverage


def _entry_metrics(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [e for e in events if e.get("type") in ("pass", "carry") and is_located(e)]
    coverage = _coverage(len(candidates), MIN_PLAYER_ENTRIES, "located passes and carries")
    if len(candidates) < MIN_PLAYER_ENTRIES:
        return {"final_third_entries_per_90": None}, {"final_third_entries_per_90": coverage}
    passes = [e for e in candidates if e.get("type") == "pass"]
    if any(e.get("completed") is None for e in passes):
        unknown = sum(1 for e in passes if e.get("completed") is None)
        return {"final_third_entries_per_90": None}, {
            "final_third_entries_per_90": _coverage(
                len(passes) - unknown,
                len(passes),
                "passes with a recorded outcome",
            )
        }
    count = sum(1 for e in candidates if is_final_third_entry(e))
    return {"final_third_entries_per_90": count}, {"final_third_entries_per_90": coverage}


def _turnover_metrics(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    actions = [
        e for e in events if e.get("type") in ("interception", "recovery", "duel") and is_located(e)
    ]
    coverage = _coverage(len(actions), MIN_PLAYER_DEF_ACTIONS, "located defensive actions")
    if len(actions) < MIN_PLAYER_DEF_ACTIONS:
        return {"high_turnovers_per_90": None}, {"high_turnovers_per_90": coverage}
    duels = [e for e in actions if e.get("type") == "duel"]
    if any(e.get("won") is None for e in duels):
        unknown = sum(1 for e in duels if e.get("won") is None)
        return {"high_turnovers_per_90": None}, {
            "high_turnovers_per_90": _coverage(
                len(duels) - unknown,
                len(duels),
                "duels with a recorded outcome",
            )
        }
    count = sum(1 for e in actions if is_high_turnover(e))
    return {"high_turnovers_per_90": count}, {"high_turnovers_per_90": coverage}


def build_player_profile(
    events: Sequence[Mapping[str, Any]],
    minutes: float,
) -> dict[str, Any]:
    """Style measurements for one player in one match.

    ``events`` is the same normalized stream the team profile consumes,
    filtered to the player. ``minutes`` must be positive; callers gate the
    30-minute floor before asking for a percentile, but the per-90 scaling
    itself is still defensive here.
    """
    minutes = float(minutes) if minutes and minutes > 0 else 0.0
    values: dict[str, Any] = {
        "passes_per_90": None,
        "progressive_passes_per_90": None,
        "final_third_entries_per_90": None,
        "high_turnovers_per_90": None,
        "pass_completion_pct": None,
    }
    coverage: dict[str, Any] = {}
    if minutes <= 0:
        return {"metrics": values, "coverage": coverage, "minutes": minutes}

    pass_metrics, pass_coverage = _pass_metrics(events)
    entry_metrics, entry_coverage = _entry_metrics(events)
    turnover_metrics, turnover_coverage = _turnover_metrics(events)
    metrics = {**values, **pass_metrics, **entry_metrics, **turnover_metrics}
    coverage = {**pass_coverage, **entry_coverage, **turnover_coverage}

    for key in (
        "passes_per_90",
        "progressive_passes_per_90",
        "final_third_entries_per_90",
        "high_turnovers_per_90",
    ):
        value = metrics.get(key)
        if value is not None:
            metrics[key] = per_90(value, minutes)
    return {"metrics": metrics, "coverage": coverage, "minutes": minutes}


def player_unavailable_reason(key: str, profile: Mapping[str, Any]) -> str:
    """Why a player style metric could not be measured, in observed counts."""
    coverage = (profile.get("coverage") or {}).get(key) or {}
    n = coverage.get("n")
    minimum = coverage.get("minimum")
    unit = coverage.get("unit") or "events"
    if n is None or minimum is None:
        return "the events for this player do not carry what this metric needs"
    return f"needs at least {minimum} {unit}, found {n}"
