"""Phase xG Breakdown — classifies shots by the possession phase that created them.

A coach needs to know "how are we creating chances?" — this module categorises
each shot based on the type of possession sequence that preceded it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PossessionPhase(Enum):
    SETTLED_POSSESSION = "settled_possession"
    TRANSITION_ATTACK = "transition_attack"
    COUNTER_ATTACK = "counter_attack"
    SET_PIECE = "set_piece"
    DIRECT_PLAY = "direct_play"
    UNKNOWN = "unknown"


@dataclass
class PhaseXgBreakdown:
    team: str
    phases: dict[str, dict] = field(default_factory=dict)
    totals: dict = field(default_factory=dict)
    opponent_phases: dict[str, dict] = field(default_factory=dict)
    phase_distribution_pct: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "phases": dict(self.phases),
            "totals": dict(self.totals),
            "opponent_phases": dict(self.opponent_phases),
            "phase_distribution_pct": dict(self.phase_distribution_pct),
        }


@dataclass
class PhaseXgReport:
    team: str
    match_id: str
    team_breakdown: PhaseXgBreakdown
    opponent_breakdown: PhaseXgBreakdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "match_id": self.match_id,
            "team_breakdown": self.team_breakdown.to_dict(),
            "opponent_breakdown": self.opponent_breakdown.to_dict(),
        }

    def summary_text(self) -> str:
        lines = [f"Phase xG Breakdown for {self.team} (Match: {self.match_id})"]
        tb = self.team_breakdown
        lines.append(
            f"  Total: {tb.totals.get('shots', 0)} shots, "
            f"{tb.totals.get('xg', 0.0):.2f} xG, "
            f"{tb.totals.get('goals', 0)} goals"
        )
        for phase_name in sorted(tb.phases):
            p = tb.phases[phase_name]
            if p.get("shots", 0) == 0:
                continue
            pct = tb.phase_distribution_pct.get(phase_name, 0.0)
            lines.append(
                f"  {phase_name.replace('_', ' ').title()}: {p['shots']} shots, "
                f"{p['xg']:.2f} xG ({pct:.1f}%)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_possession_chains(events: list[dict]) -> list[list[int]]:
    """Group event indices into possession chains based on team changes.

    Each chain is a list of event indices that belong to the same team
    in sequence.  Shots are included in their team's chain.
    """
    if not events:
        return []

    sorted_indices = sorted(range(len(events)), key=lambda i: events[i].get("timestamp", 0.0))
    chains: list[list[int]] = []
    current_chain: list[int] = []
    current_team: str | None = None

    for idx in sorted_indices:
        team = events[idx].get("team", "")
        if not team:
            if current_chain:
                current_chain.append(idx)
            continue

        if team != current_team and current_team is not None:
            if current_chain:
                chains.append(current_chain)
            current_chain = []
        current_team = team
        current_chain.append(idx)

    if current_chain:
        chains.append(current_chain)

    return chains


def _find_chain_for_shot(shot_idx: int, chains: list[list[int]]) -> list[int] | None:
    """Return the possession chain (list of event indices) that contains *shot_idx*."""
    for chain in chains:
        if shot_idx in chain:
            return chain
    return None


def _chain_events(events: list[dict], chain_indices: list[int]) -> list[dict]:
    return [events[i] for i in chain_indices]


def _measure_chain(events: list[dict], chain_indices: list[int]) -> tuple[int, float]:
    chain_evs = _chain_events(events, chain_indices)
    passes = sum(1 for e in chain_evs if e.get("type") == "pass")
    timestamps = [e.get("timestamp", 0.0) for e in chain_evs]
    duration = max(timestamps) - min(timestamps) if len(chain_evs) > 1 else 0.001
    return passes, max(0.001, duration)


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------


def classify_possession_phase(
    events: list[dict],
    shot_index: int,
    possession_chain: list[dict],
    set_piece_events: set[int],
) -> PossessionPhase:
    """Classify the possession phase that led to a shot.

    Parameters
    ----------
    events : list[dict]
        Full match event list (used to resolve set-piece indices).
    shot_index : int
        Index of the shot event in *events*.
    possession_chain : list[dict]
        The subset of events forming the possession chain that ended in the shot.
    set_piece_events : set[int]
        Indices of set-piece events (corners, free kicks, throw-ins, etc.).

    Returns
    -------
    PossessionPhase
    """
    # 1. Set piece — shot follows a set piece within 3 events (check first
    #    so empty chain doesn't shadow set-piece classification)
    for sp_idx in sorted(set_piece_events, reverse=True):
        if sp_idx < shot_index <= sp_idx + 3:
            return PossessionPhase.SET_PIECE

    if not possession_chain:
        return PossessionPhase.UNKNOWN

    pass_count = sum(1 for e in possession_chain if e.get("type") == "pass")
    timestamps = [e.get("timestamp", 0.0) for e in possession_chain]
    duration = max(timestamps) - min(timestamps) if len(possession_chain) > 1 else 0.001

    first_ev = possession_chain[0]
    start_x = first_ev.get("start_x", 0.0)

    # 2. Direct play — first pass is a long ball (>30 m) aimed at final third
    for ev in possession_chain:
        if ev.get("type") == "pass":
            sx = ev.get("start_x", 0.0)
            ex = ev.get("end_x", 0.0)
            if abs(ex - sx) > 30.0 and ex > 70.0:
                return PossessionPhase.DIRECT_PLAY

    # 3. Short / fast attacks from own half
    if pass_count < 5 and duration < 10.0:
        if start_x < 35.0:
            return PossessionPhase.COUNTER_ATTACK
        if start_x < 52.5:
            return PossessionPhase.TRANSITION_ATTACK

    # 4. Longer possession sequences
    if pass_count >= 5 or duration >= 10.0:
        return PossessionPhase.SETTLED_POSSESSION

    return PossessionPhase.UNKNOWN


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _build_phase_breakdown(
    side_events: list[dict],
    events: list[dict],
    chains: list[list[int]],
    set_piece_indices: set[int],
) -> PhaseXgBreakdown:
    """Build a PhaseXgBreakdown for one side.

    *side_events* is the filtered list of events for the team (or opponent).
    The full *events* + *chains* are used for context classification.
    """
    phase_data: dict[str, dict] = {}
    for phase in PossessionPhase:
        phase_data[phase.value] = {
            "shots": 0,
            "goals": 0,
            "xg": 0.0,
            "shots_on_target": 0,
            "avg_xg_per_shot": 0.0,
        }

    for _shot_idx, shot_ev in enumerate(side_events):
        if shot_ev.get("type") != "shot":
            continue
        # Find this shot's index in the full events list
        full_idx = -1
        for i, ev in enumerate(events):
            if ev is shot_ev:
                full_idx = i
                break
        if full_idx == -1:
            # Fallback: try matching by id/timestamp
            for i, ev in enumerate(events):
                if ev.get("timestamp") == shot_ev.get("timestamp") and ev.get("type") == "shot":
                    full_idx = i
                    break
        if full_idx == -1:
            continue

        chain_indices = _find_chain_for_shot(full_idx, chains)
        chain_evs = _chain_events(events, chain_indices) if chain_indices else []

        phase = classify_possession_phase(events, full_idx, chain_evs, set_piece_indices)
        pv = phase.value
        phase_data[pv]["shots"] += 1
        phase_data[pv]["goals"] += 1 if shot_ev.get("is_goal") else 0
        phase_data[pv]["xg"] += shot_ev.get("xg", 0.0)
        if shot_ev.get("on_target"):
            phase_data[pv]["shots_on_target"] += 1

    for pv in phase_data:
        s = phase_data[pv]["shots"]
        phase_data[pv]["avg_xg_per_shot"] = round(phase_data[pv]["xg"] / s, 4) if s else 0.0

    totals = {"shots": 0, "goals": 0, "xg": 0.0}
    for pv in phase_data:
        totals["shots"] += phase_data[pv]["shots"]
        totals["goals"] += phase_data[pv]["goals"]
        totals["xg"] += phase_data[pv]["xg"]
    totals["xg"] = round(totals["xg"], 4)

    dist: dict[str, float] = {}
    for pv in phase_data:
        dist[pv] = round(phase_data[pv]["xg"] / totals["xg"] * 100, 1) if totals["xg"] > 0 else 0.0

    return PhaseXgBreakdown(
        team=side_events[0].get("team", "unknown") if side_events else "unknown",
        phases=phase_data,
        totals=totals,
        opponent_phases={},
        phase_distribution_pct=dist,
    )


def compute_phase_xg(
    team_events: list[dict],
    opponent_events: list[dict],
    events: list[dict],
    set_piece_event_types: set[str] | None = None,
    possession_chain_events: list[list[int]] | None = None,
) -> PhaseXgReport:
    """Compute phase-by-phase xG breakdown for a team and its opponent.

    Parameters
    ----------
    team_events : list[dict]
        Events belonging to the analysed team.
    opponent_events : list[dict]
        Events belonging to the opponent.
    events : list[dict]
        Full match event list (used for chain detection & index matching).
    set_piece_event_types : set[str], optional
        Event type strings that count as set pieces.
    possession_chain_events : list[list[int]], optional
        Pre-computed possession chains (lists of event indices).

    Returns
    -------
    PhaseXgReport
    """
    if set_piece_event_types is None:
        set_piece_event_types = {"corner", "free_kick", "throw_in", "penalty"}

    if possession_chain_events is None:
        chains = _detect_possession_chains(events)
    else:
        chains = possession_chain_events

    set_piece_indices = {
        i for i, ev in enumerate(events) if ev.get("type", "").lower() in set_piece_event_types
    }

    team_breakdown = _build_phase_breakdown(team_events, events, chains, set_piece_indices)
    opp_breakdown = _build_phase_breakdown(opponent_events, events, chains, set_piece_indices)

    team_id = team_events[0].get("team", "home") if team_events else "home"

    return PhaseXgReport(
        team=team_id,
        match_id=events[0].get("match_id", "") if events else "",
        team_breakdown=team_breakdown,
        opponent_breakdown=opp_breakdown,
    )
