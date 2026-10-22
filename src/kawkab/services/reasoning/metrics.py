"""Event-stat precomputation and metric extraction for the reasoning engine.

Two responsibilities:

1. ``precompute_event_stats`` — the single pass over a pool of events
   that all checkers share (the engine's original fixed 16-key dict,
   extended with the buckets the new pattern checkers need).
2. ``extract_metric`` — hypothesis-level evidence lookup. Rule YAML
   hypothesis blocks declare ``evidence_required`` entries like
   ``- {metric: offside_line_breaks, threshold: ">3"}``. The metrics
   namespace overlaps the checker evidence namespace wherever the rule
   author named a concrete quantity, so ``extract_metric`` reads the
   checker's evidence dict first, then falls back to dedicated
   event_stats buckets. Unknown metrics return ``None`` — never a
   fabricated number.
"""

from __future__ import annotations

import re
from typing import Any

_THRESHOLD_RE = re.compile(r"^\s*(>=|<=|>|<)?\s*(-?\d+(?:\.\d+)?)\s*(m|s|%|m/s)?\s*$")


def precompute_event_stats(events: list[dict]) -> dict:
    """Single-pass bucketing of a (possibly multi-match) event pool.

    Returns the original 16-key stats dict plus the extended buckets
    the new checkers read. Pure function; no IO.
    """
    stats: dict[str, Any] = {
        # original buckets (contract kept: existing tests depend on keys)
        "goals": [],
        "turnovers": [],
        "shots": [],
        "passes": [],
        "crosses": [],
        "set_piece_goals": [],
        "counter_attack_shots": [],
        "counter_attack_goals": [],
        "final_third_events": [],
        "behind_def_line_events": [],
        "1v1_situations": [],
        "striker_passes": [],
        "own_half_turnovers": [],
        "late_events": [],
        "first_events": [],
        "total_events": 0,
        # extended buckets for the new pattern checkers
        "corners": [],
        "dribbles": [],
        "fouls": [],
        "own_half_fouls": [],
        "offsides_won": [],
        "offside_trap_attempts": [],
        "offside_trap_failures": [],
        "aerial_duels": [],
        "key_passes": [],
        "left_side_events": [],
        "right_side_events": [],
        "zone_gap_receptions": [],
        "opposition_transitions": [],
        "press_sequences": [],
        "recovery_runs": [],
        "defensive_shape_breaks": [],
        "winger_defensive_actions": [],
        "striker_press_attempts": [],
        "midfield_gaps": [],
        "gk_distributions": [],
        "fullback_attacking_actions": [],
        "cb_positioning_errors": [],
        "opponent_shots_after_our_loss": [],
        "total_duration": 0.0,
    }
    if not events:
        return stats

    total_duration = 0.0
    timestamps = [e.get("timestamp", 0) for e in events if isinstance(e, dict)]
    if timestamps:
        total_duration = float(max(timestamps))
    stats["total_duration"] = total_duration

    for ev in events:
        if not isinstance(ev, dict):
            continue
        stats["total_events"] += 1
        etype = str(ev.get("type") or ev.get("event_type") or "")
        zone = ev.get("zone") or ""
        timestamp = ev.get("timestamp", 0)
        situation = ev.get("situation") or ""
        outcome = ev.get("outcome") or ""
        side = ev.get("side") or ev.get("flank") or ""

        if etype == "goal":
            stats["goals"].append(ev)
            if situation in ("corner", "free_kick", "throw_in"):
                stats["set_piece_goals"].append(ev)
        elif etype == "turnover":
            stats["turnovers"].append(ev)
            if zone in ("defensive_third", "middle_third"):
                stats["own_half_turnovers"].append(ev)
        elif etype == "shot":
            stats["shots"].append(ev)
            if situation == "counter_attack":
                stats["counter_attack_shots"].append(ev)
                if outcome == "goal":
                    stats["counter_attack_goals"].append(ev)
        elif etype == "pass":
            stats["passes"].append(ev)
            if ev.get("to_position") == "striker":
                stats["striker_passes"].append(ev)
            if ev.get("key_pass") or ev.get("assist"):
                stats["key_passes"].append(ev)
        elif etype == "cross":
            stats["crosses"].append(ev)
        elif etype == "1v1_situation":
            stats["1v1_situations"].append(ev)
        elif etype == "corner":
            stats["corners"].append(ev)
        elif etype == "dribble":
            stats["dribbles"].append(ev)
        elif etype == "foul":
            stats["fouls"].append(ev)
            if zone in ("defensive_third", "middle_third"):
                stats["own_half_fouls"].append(ev)
        elif etype == "offside":
            # We won the offside trap if the offender is the opponent.
            if str(ev.get("team", "")) in ("away", "opponent"):
                stats["offsides_won"].append(ev)

        for bucket, flag in (
            ("offside_trap_attempts", "trap_attempt"),
            ("offside_trap_failures", "trap_failed"),
            ("aerial_duels", None),
            ("zone_gap_receptions", None),
            ("opposition_transitions", None),
            ("press_sequences", None),
            ("recovery_runs", None),
            ("defensive_shape_breaks", None),
            ("winger_defensive_actions", None),
            ("striker_press_attempts", None),
            ("midfield_gaps", None),
            ("gk_distributions", None),
            ("fullback_attacking_actions", None),
            ("cb_positioning_errors", None),
            ("opponent_shots_after_our_loss", None),
        ):
            if flag is not None:
                if ev.get(flag):
                    stats[bucket].append(ev)
            elif etype == bucket:  # not reachable: buckets use explicit matching below
                stats[bucket].append(ev)

        # explicit matching for buckets that ride on event_type variants
        if etype == "aerial_duel":
            stats["aerial_duels"].append(ev)
        elif etype == "reception" and (ev.get("gap") or ev.get("between_lines")):
            stats["zone_gap_receptions"].append(ev)
        elif etype == "transition":
            if str(ev.get("team", "")) in ("away", "opponent"):
                stats["opposition_transitions"].append(ev)
        elif etype == "press_sequence":
            stats["press_sequences"].append(ev)
        elif etype == "recovery_run":
            stats["recovery_runs"].append(ev)
        elif etype == "shape_break":
            stats["defensive_shape_breaks"].append(ev)
        elif etype == "press_attempt" and ev.get("position") == "striker":
            stats["striker_press_attempts"].append(ev)
        elif etype == "midfield_gap":
            stats["midfield_gaps"].append(ev)
        elif etype == "gk_distribution":
            stats["gk_distributions"].append(ev)
        elif etype == "cb_positioning_error":
            stats["cb_positioning_errors"].append(ev)
        elif (
            etype == "shot"
            and str(ev.get("team", "")) in ("away", "opponent")
            and ev.get("within_10s_of_our_loss")
        ):
            stats["opponent_shots_after_our_loss"].append(ev)

        # fullback/winger action attribution by position
        pos = str(ev.get("position") or "")
        if (
            zone == "final_third"
            and pos in ("fullback", "left_back", "right_back")
            and etype in ("cross", "pass", "dribble")
            and (ev.get("overlap") or ev.get("attacking"))
        ):
            stats["fullback_attacking_actions"].append(ev)
        if pos in ("winger", "left_winger", "right_winger") and etype in (
            "tackle",
            "interception",
            "recovery_run",
        ):
            stats["winger_defensive_actions"].append(ev)

        if side == "left":
            stats["left_side_events"].append(ev)
        elif side == "right":
            stats["right_side_events"].append(ev)

        if zone == "final_third":
            stats["final_third_events"].append(ev)
        if zone == "behind_defensive_line":
            stats["behind_def_line_events"].append(ev)

        if total_duration > 0:
            q1 = total_duration * 0.25
            q3 = total_duration * 0.75
            if timestamp > q3:
                stats["late_events"].append(ev)
            elif timestamp < q1:
                stats["first_events"].append(ev)

    return stats


# ----------------------------------------------------------------------
# Hypothesis evidence extraction
# ----------------------------------------------------------------------


def _parse_threshold(raw: Any) -> tuple[str, float] | None:
    """``">3"`` -> (\">\", 3.0); ``\"<-15m\"`` -> (\"<\", -15.0)."""
    if isinstance(raw, (int, float)):
        return (">=", float(raw))
    if not isinstance(raw, str):
        return None
    m = _THRESHOLD_RE.match(raw)
    if not m:
        return None
    op = m.group(1) or ">="
    return (op, float(m.group(2)))


def parse_threshold(raw: Any) -> tuple[str, float] | None:
    """Public threshold parsing for callers outside the engine (efficacy)."""
    return _parse_threshold(raw)


def _satisfies(value: float, op: str, ref: float) -> bool:
    if op == ">":
        return value > ref
    if op == "<":
        return value < ref
    if op == ">=":
        return value >= ref
    if op == "<=":
        return value <= ref
    return False


def extract_metric(
    metric: str,
    evidence: dict[str, Any],
    event_stats: dict[str, Any],
) -> float | None:
    """Resolve a rule-declared metric name to a concrete number.

    Order: checker evidence keys first (they are the observed
    quantities for the matched pattern), then dedicated event_stats
    buckets. Returns None when the metric was not observed — the caller
    must treat that as "unknown", not 0.
    """
    if metric in evidence and isinstance(evidence[metric], (int, float)):
        return float(evidence[metric])

    # explicit bucket lookups for metrics that name a bucket
    bucket_map: dict[str, str] = {
        "turnovers_in_defensive_third": "turnovers",
        "offsides_won": "offsides_won",
        "offside_line_breaks": "behind_def_line_events",
        "goals_from_through_balls": "goals",
        "goals_conceded_from_through_balls": "goals",
        "aerial_duels_lost": "aerial_duels",
        "unmarked_attackers_in_box": "zone_gap_receptions",
        "striker_pressure_attempts": "striker_press_attempts",
        "striker_isolated_press_attempts": "striker_press_attempts",
        "gk_distribution_errors": "gk_distributions",
        "passes_between_lines": "zone_gap_receptions",
        "receptions_between_lines": "zone_gap_receptions",
        "recovery_runs_not_direct": "recovery_runs",
        "times_defensive_shape_broken": "defensive_shape_breaks",
        "defensive_shape_breaks": "defensive_shape_breaks",
        "set_piece_confusion": "set_piece_goals",
        "second_ball_recovery_rate": "press_sequences",
        "opponent_shots_within_8s_of_loss_pct": "opponent_shots_after_our_loss",
        "opponent_shots_within_10s_of_loss": "opponent_shots_after_our_loss",
        "winger_reaction_time_to_loss": "recovery_runs",
        "winger_distance_from_defensive_position": "midfield_gaps",
        "midfielder_avg_position_x": "midfield_gaps",
        "midfield_vertical_spacing": "midfield_gaps",
        "gk_avg_position_x": "gk_distributions",
        "team_width_after_loss_m": "recovery_runs",
        "max_speed_last_15_min": "late_events",
    }
    bucket = bucket_map.get(metric)
    if bucket and isinstance(event_stats.get(bucket), list):
        return float(len(event_stats[bucket]))

    # generic snake_case fallbacks onto checker evidence
    snake = metric.replace("avg_", "").replace("total_", "")
    for key in (metric, snake):
        val = evidence.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def evaluate_hypothesis(
    hypothesis: dict[str, Any],
    evidence: dict[str, Any],
    event_stats: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one rule hypothesis against observed evidence.

    Returns a result dict with provenance::

        {
            "hypothesis_id": ...,
            "weight": 0.4,
            "confidence_boost": 0.35,
            "evaluated": True,          # all metrics were observable
            "criteria": [
                {"metric": ..., "threshold": ">3", "observed": 5.0,
                 "met": True, "known": True}, ...
            ],
            "met_all": True,
            "boost_applied": 0.35,      # 0.0 unless met_all
        }

    A hypothesis whose metrics are not observable is reported as
    ``evaluated: False`` with ``boost_applied: 0.0`` — never silently
    treated as failed, and never granted a boost it did not earn.
    """
    raw_reqs = hypothesis.get("evidence_required") or []
    criteria: list[dict[str, Any]] = []
    known_all = True

    for req in raw_reqs:
        if not isinstance(req, dict):
            continue
        metric = str(req.get("metric") or "")
        threshold_raw = req.get("threshold")
        parsed = _parse_threshold(threshold_raw)
        observed = extract_metric(metric, evidence, event_stats)
        known = parsed is not None and observed is not None
        if parsed is not None and observed is not None:
            met = _satisfies(float(observed), parsed[0], parsed[1])
        else:
            met = False
        if not known:
            known_all = False
        criteria.append(
            {
                "metric": metric,
                "threshold": str(threshold_raw),
                "observed": observed,
                "met": met,
                "known": known,
            }
        )

    met_all = bool(criteria) and all(c["met"] for c in criteria)
    boost = float(hypothesis.get("confidence_boost") or 0.0)
    return {
        "hypothesis_id": hypothesis.get("hypothesis_id") or hypothesis.get("id"),
        "weight": float(hypothesis.get("weight") or 0.0),
        "confidence_boost": boost,
        "evaluated": known_all,
        "criteria": criteria,
        "met_all": met_all,
        "boost_applied": boost if met_all else 0.0,
    }
