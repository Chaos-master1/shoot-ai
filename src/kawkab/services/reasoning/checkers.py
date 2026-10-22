"""Per-pattern confidence checkers for the reasoning engine.

One function per ``pattern_signature.type`` in the knowledge-base rule
files. Each checker returns ``(confidence, evidence)`` where confidence
is a base match strength in [0, 1] and evidence is a flat dict of the
concrete quantities that produced it (persisted verbatim with each
diagnosis for provenance).

Checkers are deliberately small and pure: analysis + precomputed event
stats in, confidence + evidence out. No KB access, no persistence, no
IO. Shared metric helpers live in ``metrics.py``.

Every checker is honest about unknown data: when the metric a rule
needs was not captured (e.g. passes with no ``completed`` field), it
returns 0.0 rather than guessing.
"""

from __future__ import annotations

from typing import Any

# Checker signature: (MatchAnalysis, dict, int) -> tuple[float, dict].
# Third arg is the number of distinct matches whose events were pooled.
Checker = Any


def _r2(v: float) -> float:
    return round(v, 2)


def _pct(part: int, total: int) -> float:
    return part / total if total else 0.0


def _completed_of(seq: list, *keys: str) -> int | None:
    """Count events marked successful, or None if no success field exists.

    A pass/cross/dribble row may carry ``completed`` (0/1/bool) or an
    ``outcome`` like "completed"/"successful". If *no* row in the pool
    carries any success field, the accuracy is unknowable — return None
    so callers can decline to fabricate a rate.
    """
    seen_success_field = False
    count = 0
    for ev in seq:
        outcome = str(ev.get("outcome") or "")
        if ev.get("completed") is not None or outcome:
            seen_success_field = True
        if ev.get("completed") or outcome in {"completed", "successful", "shot_created"}:
            count += 1
    return count if seen_success_field else None


# ---------------------------------------------------------------------
# Offensive
# ---------------------------------------------------------------------


def check_low_passing_accuracy(analysis, event_stats, match_count=1):
    """Completion rate well under professional baseline (>=70% match)."""
    passes = event_stats.get("passes") or []
    if len(passes) < 20:
        return 0.0, {}
    completed = _completed_of(passes)
    if completed is None:
        return 0.0, {}
    acc = _pct(completed, len(passes))
    if acc < 0.70:
        return min(0.75, 0.5 + (0.70 - acc)), {
            "pass_accuracy": _r2(acc),
            "passes": len(passes),
            "completed": completed,
        }
    return 0.0, {}


def check_low_shot_conversion(analysis, event_stats, match_count=1):
    """Shots-per-goal far above the ~10:1 professional baseline.

    Needs enough volume to mean anything: 6 shots per pooled match.
    """
    shots = event_stats.get("shots") or []
    if len(shots) < 6 * max(1, match_count):
        return 0.0, {}
    goals = len(event_stats.get("goals") or [])
    conv = _pct(goals, len(shots))
    if conv < 0.10:
        return min(0.70, 0.5 + (0.10 - conv)), {
            "shot_conversion": _r2(conv),
            "shots": len(shots),
            "goals": goals,
        }
    return 0.0, {}


def check_poor_corner_kicks(analysis, event_stats, match_count=1):
    """Many corners, few shots or goals directly from them."""
    corners = event_stats.get("corners") or []
    if len(corners) < 5:
        return 0.0, {}
    productive = sum(1 for c in corners if str(c.get("outcome") or "") in {"shot", "goal"})
    conv = _pct(productive, len(corners))
    if conv < 0.20:
        return 0.6, {
            "corners": len(corners),
            "corners_producing_shot_or_goal": productive,
            "conversion": _r2(conv),
        }
    return 0.0, {}


def check_low_dribble_success(analysis, event_stats, match_count=1):
    """Dribbles attempted with a poor success rate."""
    dribbles = event_stats.get("dribbles") or []
    if not dribbles:
        return 0.0, {}
    won = _completed_of(dribbles)
    if won is None:
        return 0.0, {}
    rate = _pct(won, len(dribbles))
    if len(dribbles) >= 8 and rate < 0.35:
        return 0.55, {
            "dribbles": len(dribbles),
            "dribbles_won": won,
            "success_rate": _r2(rate),
        }
    return 0.0, {}


def check_low_vertical_progression(analysis, event_stats, match_count=1):
    """Possession that never accelerates toward goal: passes dominate,
    forward passes rare, few entries into the final third."""
    passes = event_stats.get("passes") or []
    ft = event_stats.get("final_third_events") or []
    if len(passes) < 40:
        return 0.0, {}
    with_direction = [p for p in passes if p.get("direction")]
    if len(with_direction) < len(passes) * 0.5:
        return 0.0, {}
    forward = sum(1 for p in with_direction if p.get("direction") == "forward")
    fwd_rate = _pct(forward, len(with_direction))
    if fwd_rate < 0.25 and len(ft) < 12 * max(1, match_count):
        return 0.55, {
            "forward_pass_rate": _r2(fwd_rate),
            "final_third_events": len(ft),
            "passes_analyzed": len(with_direction),
        }
    return 0.0, {}


def check_high_possession_low_progression(analysis, event_stats, match_count=1):
    """High possession share that produces little: many passes, few shots."""
    passes = event_stats.get("passes") or []
    shots = event_stats.get("shots") or []
    if len(passes) < 80 * max(1, match_count):
        return 0.0, {}
    if analysis.home_team.possession_pct < 55 and analysis.away_team.possession_pct < 55:
        return 0.0, {}
    shots_per_100 = 100.0 * len(shots) / len(passes)
    if shots_per_100 < 4:
        return 0.6, {
            "passes": len(passes),
            "shots": len(shots),
            "shots_per_100_passes": _r2(shots_per_100),
            "possession_pct": _r2(analysis.home_team.possession_pct),
        }
    return 0.0, {}


def check_lopsided_attack(analysis, event_stats, match_count=1):
    """Attack concentrated overwhelmingly on one flank."""
    left = event_stats.get("left_side_events") or []
    right = event_stats.get("right_side_events") or []
    total = len(left) + len(right)
    if total < 20:
        return 0.0, {}
    share = max(len(left), len(right)) / total
    if share >= 0.65:
        return 0.6, {
            "left_side_events": len(left),
            "right_side_events": len(right),
            "dominant_side_share": _r2(share),
        }
    return 0.0, {}


def check_low_creative_actions(analysis, event_stats, match_count=1):
    """Almost nothing manufactured for others: key passes, through balls,
    dribbles that beat a man — weighted per match.

    Absence of creativity is only claimable against an activity
    baseline: with fewer than ~30 passing/shot events per pooled match
    there is not enough possession to expect creative actions, and an
    empty or thin event pool must never produce this diagnosis.
    """
    passes = event_stats.get("passes") or []
    dribbles = event_stats.get("dribbles") or []
    shots = event_stats.get("shots") or []
    activity = len(passes) + len(shots)
    if activity < 30 * max(1, match_count):
        return 0.0, {}
    key = event_stats.get("key_passes") or []
    creative = len(key) + sum(1 for d in dribbles if d.get("outcome") == "beaten")
    if creative < 4 * max(1, match_count):
        return 0.55, {
            "key_passes": len(key),
            "successful_takeons": sum(1 for d in dribbles if d.get("outcome") == "beaten"),
            "creative_actions": creative,
            "activity_baseline": activity,
        }
    return 0.0, {}


# ---------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------


def check_reception_in_zone_gap(analysis, event_stats, match_count=1):
    """Opponents receiving between our lines/units unmarked."""
    receptions = event_stats.get("zone_gap_receptions") or []
    if len(receptions) >= 4 * max(1, match_count):
        unmarked = sum(1 for r in receptions if not r.get("marked"))
        if unmarked >= len(receptions) * 0.5:
            return 0.65, {
                "zone_gap_receptions": len(receptions),
                "unmarked": unmarked,
            }
    return 0.0, {}


def check_aerial_duel_loss_defensive(analysis, event_stats, match_count=1):
    """Defensive aerial duels lost at a damaging rate."""
    duels = event_stats.get("aerial_duels") or []
    if len(duels) < 10:
        return 0.0, {}
    won = _completed_of(duels)
    if won is None:
        return 0.0, {}
    rate = _pct(won, len(duels))
    if rate < 0.45:
        return 0.6, {
            "aerial_duels": len(duels),
            "duels_won": won,
            "win_rate": _r2(rate),
        }
    return 0.0, {}


def check_poor_defensive_transition(analysis, event_stats, match_count=1):
    """Shape still broken after losing the ball: opposition progresses
    quickly from our turnovers."""
    transitions = event_stats.get("opposition_transitions") or []
    if len(transitions) < 3 * max(1, match_count):
        return 0.0, {}
    fast = sum(1 for t in transitions if t.get("shot_within_10s") or t.get("progressed_fast"))
    if fast >= len(transitions) * 0.4:
        return 0.65, {
            "opposition_transitions": len(transitions),
            "fast_transitions": fast,
        }
    return 0.0, {}


def check_press_broken_by_simple_pass(analysis, event_stats, match_count=1):
    """Press beaten: opponents escape pressure with one or two passes."""
    press_sequences = event_stats.get("press_sequences") or []
    if len(press_sequences) < 6 * max(1, match_count):
        return 0.0, {}
    broken = sum(1 for s in press_sequences if s.get("beaten_by_simple_pass"))
    if broken >= len(press_sequences) * 0.4:
        return 0.6, {
            "press_sequences": len(press_sequences),
            "broken_by_simple_pass": broken,
        }
    return 0.0, {}


def check_high_foul_rate(analysis, event_stats, match_count=1):
    """Fouls per match well above professional norms (~12-14)."""
    fouls = event_stats.get("fouls") or []
    per_match = len(fouls) / max(1, match_count)
    if per_match >= 15:
        return 0.55, {
            "fouls": len(fouls),
            "fouls_per_match": _r2(per_match),
            "own_half_fouls": len(event_stats.get("own_half_fouls") or []),
        }
    return 0.0, {}


def check_turnover_in_defensive_third(analysis, event_stats, match_count=1):
    """Ball lost in our defensive third at dangerous frequency."""
    turnovers = event_stats.get("turnovers") or []
    own = event_stats.get("own_half_turnovers") or []
    def_third = [t for t in turnovers if t.get("zone") == "defensive_third"]
    per_match = len(def_third) / max(1, match_count)
    if per_match >= 6:
        return 0.6, {
            "defensive_third_turnovers": len(def_third),
            "total_turnovers": len(turnovers),
            "own_half_turnovers": len(own),
            "per_match": _r2(per_match),
        }
    return 0.0, {}


def check_low_compactness(analysis, event_stats, match_count=1):
    """Defensive block stretched: opponents receive between the lines
    and central space is repeatedly used."""
    receptions = event_stats.get("zone_gap_receptions") or []
    central = [r for r in receptions if r.get("zone") in {"central", "between_lines"}]
    if len(central) >= 5 * max(1, match_count):
        return 0.6, {
            "central_receptions_between_lines": len(central),
            "zone_gap_receptions": len(receptions),
        }
    return 0.0, {}


def check_low_offside_pressure(analysis, event_stats, match_count=1):
    """Almost no offside traps attempted: line never steps."""
    offsides = event_stats.get("offsides_won") or []
    if len(offsides) < 1 * max(1, match_count):
        behind = event_stats.get("behind_def_line_events") or []
        if len(behind) >= 3 * max(1, match_count):
            return 0.5, {
                "offsides_won": len(offsides),
                "balls_behind_defense": len(behind),
            }
    return 0.0, {}


def check_offside_trap_ineffective(analysis, event_stats, match_count=1):
    """Trap attempted repeatedly but beaten — opponents still get behind."""
    offsides = event_stats.get("offsides_won") or []
    attempts = event_stats.get("offside_trap_attempts") or []
    behind = event_stats.get("behind_def_line_events") or []
    if len(attempts) >= 3 and len(behind) >= 3 * max(1, match_count):
        return 0.55, {
            "offside_trap_attempts": len(attempts),
            "offsides_won": len(offsides),
            "balls_behind_defense": len(behind),
        }
    return 0.0, {}


def check_offside_failed(analysis, event_stats, match_count=1):
    """Trap actively failing: more opponents played onside than caught."""
    attempts = event_stats.get("offside_trap_attempts") or []
    broken = event_stats.get("offside_trap_failures") or []
    if len(attempts) >= 4 and len(broken) > len(attempts) * 0.5:
        return 0.6, {
            "offside_trap_attempts": len(attempts),
            "offside_trap_failures": len(broken),
        }
    return 0.0, {}


# ---------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------


def check_no_pressure_after_loss(analysis, event_stats, match_count=1):
    """Losses met with nothing: opponents play forward unpressured."""
    losses = event_stats.get("own_half_turnovers") or event_stats.get("turnovers") or []
    countered = event_stats.get("opponent_shots_after_our_loss") or []
    if len(losses) < 10 * max(1, match_count):
        return 0.0, {}
    if len(countered) >= 3 * max(1, match_count):
        return 0.6, {
            "turnovers": len(losses),
            "opponent_shots_within_10s_of_loss": len(countered),
        }
    return 0.0, {}


def check_slow_recovery_after_loss(analysis, event_stats, match_count=1):
    """Recovery runs slow or misdirected after losing the ball."""
    recoveries = event_stats.get("recovery_runs") or []
    if not recoveries:
        return 0.0, {}
    fast = sum(1 for r in recoveries if r.get("direct") and r.get("fast"))
    rate = _pct(fast, len(recoveries))
    if rate < 0.4:
        return 0.55, {
            "recovery_runs": len(recoveries),
            "direct_and_fast": fast,
            "fast_rate": _r2(rate),
        }
    return 0.0, {}


# ---------------------------------------------------------------------
# Meta / individual
# ---------------------------------------------------------------------


def check_late_goals_conceded(analysis, event_stats, match_count=1):
    """Goals conceded disproportionately in the final 15 minutes."""
    goals = event_stats.get("goals") or []
    late = [g for g in goals if _is_late(g, event_stats)]
    if len(goals) >= 2 and len(late) >= max(2, match_count):
        return 0.6, {
            "late_goals": len(late),
            "total_goals": len(goals),
        }
    return 0.0, {}


def check_defensive_disorganization(analysis, event_stats, match_count=1):
    """Shape broken repeatedly: goals conceded from disorganized states."""
    breaks = event_stats.get("defensive_shape_breaks") or []
    if len(breaks) >= 4 * max(1, match_count):
        return 0.6, {
            "defensive_shape_breaks": len(breaks),
            "goals_from_breaks": sum(1 for b in breaks if b.get("led_to_goal")),
        }
    return 0.0, {}


def check_winger_not_tracking_back(analysis, event_stats, match_count=1):
    """Wingers absent from defensive actions on their flank.

    Absence is only claimable against an activity baseline: fewer than
    ~40 events per pooled match says nothing about winger work rate.
    """
    total = event_stats.get("total_events") or 0
    if total < 40 * max(1, match_count):
        return 0.0, {}
    flank_defs = event_stats.get("winger_defensive_actions") or []
    per_match = len(flank_defs) / max(1, match_count)
    if per_match < 2:
        return 0.55, {
            "winger_defensive_actions": len(flank_defs),
            "per_match": _r2(per_match),
            "activity_baseline": total,
        }
    return 0.0, {}


def check_striker_passive_press(analysis, event_stats, match_count=1):
    """Striker presses rarely when possession is lost up front."""
    attempts = event_stats.get("striker_press_attempts") or []
    losses = event_stats.get("final_third_events") or []
    if not attempts and len(losses) >= 8 * max(1, match_count):
        return 0.5, {
            "striker_press_attempts": 0,
            "opposition_final_third_touches_observed": len(losses),
        }
    return 0.0, {}


def check_midfield_disconnected(analysis, event_stats, match_count=1):
    """Midfield unit far from both lines: big vertical gaps."""
    gaps = event_stats.get("midfield_gaps") or []
    if len(gaps) >= 5 * max(1, match_count):
        large = sum(1 for g in gaps if (g.get("vertical_gap_m") or 0) > 15)
        if large >= len(gaps) * 0.5:
            return 0.6, {
                "midfield_gap_samples": len(gaps),
                "gaps_over_15m": large,
            }
    return 0.0, {}


def check_gk_distribution_loss(analysis, event_stats, match_count=1):
    """Goalkeeper distribution repeatedly surrenders possession."""
    dists = event_stats.get("gk_distributions") or []
    if len(dists) < 6 * max(1, match_count):
        return 0.0, {}
    errors = _completed_of(dists)
    if errors is None:
        return 0.0, {}
    err_rate = 1.0 - _pct(errors, len(dists))
    if err_rate > 0.35:
        return 0.6, {
            "gk_distributions": len(dists),
            "distribution_error_rate": _r2(err_rate),
        }
    return 0.0, {}


def check_fullback_not_attacking(analysis, event_stats, match_count=1):
    """Fullbacks offer nothing in attack: no overlaps, few final-third events.

    Absence is only claimable against an activity baseline: fewer than
    ~60 events per pooled match says nothing about fullback involvement.
    (An empty match must produce zero diagnoses, not this one.)
    """
    total = event_stats.get("total_events") or 0
    if total < 60 * max(1, match_count):
        return 0.0, {}
    fb_attacks = event_stats.get("fullback_attacking_actions") or []
    per_match = len(fb_attacks) / max(1, match_count)
    if per_match < 2:
        return 0.5, {
            "fullback_attacking_actions": len(fb_attacks),
            "per_match": _r2(per_match),
            "activity_baseline": total,
        }
    return 0.0, {}


def check_cb_positioning_error(analysis, event_stats, match_count=1):
    """Center backs repeatedly caught out of position."""
    errors = event_stats.get("cb_positioning_errors") or []
    if len(errors) >= 3 * max(1, match_count):
        return 0.6, {
            "cb_positioning_errors": len(errors),
            "led_to_shots": sum(1 for e in errors if e.get("led_to_shot")),
        }
    return 0.0, {}


def check_low_aerial_win_rate(analysis, event_stats, match_count=1):
    """General aerial weakness across all duels."""
    duels = event_stats.get("aerial_duels") or []
    if len(duels) < 12:
        return 0.0, {}
    won = _completed_of(duels)
    if won is None:
        return 0.0, {}
    rate = _pct(won, len(duels))
    if rate < 0.45:
        return 0.55, {
            "aerial_duels": len(duels),
            "duels_won": won,
            "win_rate": _r2(rate),
        }
    return 0.0, {}


def _is_late(ev: dict, event_stats: dict) -> bool:
    """Final ~15 minutes, or final quarter when duration is unknown."""
    ts = ev.get("timestamp") or 0
    duration = event_stats.get("total_duration") or 0
    if duration:
        return ts >= duration - 15 * 60
    return bool(ev.get("minute") and int(ev["minute"]) >= 75)
