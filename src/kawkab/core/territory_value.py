"""Territory Compounding — accumulates xT threat across possession chains.

Measures territorial effectiveness by tracking how much xT a team generates
from each pitch zone, and how much it concedes, providing a spatial map
of where the game was won or lost.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from kawkab.core.xt_model import ExpectedThreatModel

if TYPE_CHECKING:

    class _XtModelProtocol(Protocol):
        """Structural subset of ExpectedThreatModel used below."""

        def compute_action_xt(self, sx: float, sy: float, ex: float, ey: float) -> float: ...

        def _zone_from_position(self, x: float, y: float) -> tuple[int, int]: ...


@dataclass
class TerritoryCell:
    zone_x: int
    zone_y: int
    xT_gained: float = 0.0
    xT_conceded: float = 0.0
    net_xT: float = 0.0
    possession_time_pct: float = 0.0
    event_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_x": self.zone_x,
            "zone_y": self.zone_y,
            "xT_gained": round(self.xT_gained, 4),
            "xT_conceded": round(self.xT_conceded, 4),
            "net_xT": round(self.net_xT, 4),
            "possession_time_pct": round(self.possession_time_pct, 1),
            "event_count": self.event_count,
        }


@dataclass
class TerritoryReport:
    team: str
    match_id: str
    cells: list[TerritoryCell] = field(default_factory=list)
    total_xT_gained: float = 0.0
    total_xT_conceded: float = 0.0
    net_territory_value: float = 0.0
    possession_chains: list[dict] = field(default_factory=list)
    dominant_zones: list[dict] = field(default_factory=list)
    territory_timeline: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "match_id": self.match_id,
            "cells": [c.to_dict() for c in self.cells],
            "total_xT_gained": round(self.total_xT_gained, 4),
            "total_xT_conceded": round(self.total_xT_conceded, 4),
            "net_territory_value": round(self.net_territory_value, 4),
            "possession_chains": self.possession_chains,
            "dominant_zones": self.dominant_zones,
            "territory_timeline": self.territory_timeline,
        }

    def summary_text(self) -> str:
        lines = [f"Territory Report for {self.team} (Match: {self.match_id})"]
        lines.append(f"  Total xT gained: {self.total_xT_gained:.3f}")
        lines.append(f"  Total xT conceded: {self.total_xT_conceded:.3f}")
        lines.append(f"  Net territory value: {self.net_territory_value:+.3f}")
        lines.append(f"  Dominant zones: {len(self.dominant_zones)}")
        lines.append(f"  Possession chains: {len(self.possession_chains)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Possession chain detection
# ---------------------------------------------------------------------------


def _detect_possession_chains_full(
    events: list[dict],
) -> list[list[int]]:
    """Group all event indices into possession chains by team."""
    if not events:
        return []

    sorted_indices = sorted(range(len(events)), key=lambda i: events[i].get("timestamp", 0.0))
    chains: list[list[int]] = []
    current: list[int] = []
    current_team: str | None = None

    for idx in sorted_indices:
        team = events[idx].get("team", "")
        if not team:
            if current:
                current.append(idx)
            continue
        if team != current_team and current_team is not None:
            if current:
                chains.append(current)
            current = []
        current_team = team
        current.append(idx)

    if current:
        chains.append(current)

    return chains


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def _make_default_xt_model(
    events: list[dict],
    grid_rows: int,
    grid_cols: int,
) -> ExpectedThreatModel:
    model = ExpectedThreatModel(rows=grid_rows, cols=grid_cols)
    model.build_transition_matrix(events)
    return model


def compute_territory_value(
    team_events: list[dict],
    opponent_events: list[dict],
    match_events: list[dict],
    team_id: str,
    xt_model: _XtModelProtocol | None = None,
    possession_chains: list[list[int]] | None = None,
    grid_rows: int = 20,
    grid_cols: int = 32,
) -> TerritoryReport:
    """Compute territory value by accumulating xT across possession chains.

    Parameters
    ----------
    team_events : list[dict]
        Events for the analysed team.
    opponent_events : list[dict]
        Events for the opponent.
    match_events : list[dict]
        Full match event list.
    team_id : str
        Team identifier.
    xt_model : ExpectedThreatModel, optional
        Pre-built xT model.  If ``None`` a default 20×32 model is built.
    possession_chains : list[list[int]], optional
        Pre-computed possession chains (lists of event indices).
    grid_rows : int
        Number of vertical zones in the xT grid.
    grid_cols : int
        Number of horizontal zones in the xT grid.

    Returns
    -------
    TerritoryReport
    """
    if not match_events or (not team_events and not opponent_events):
        return TerritoryReport(team=team_id, match_id="")

    match_id = match_events[0].get("match_id", "") if match_events else ""

    if xt_model is None:
        xt_model = _make_default_xt_model(match_events, grid_rows, grid_cols)
    assert xt_model is not None  # narrowed for mypy (optional-arg default built above)

    if possession_chains is None:
        possession_chains = _detect_possession_chains_full(match_events)

    # Map team id to a boolean flag
    team_team = team_events[0].get("team", "") if team_events else ""

    # Zone accumulation
    gained: dict[tuple[int, int], float] = defaultdict(float)
    conceded: dict[tuple[int, int], float] = defaultdict(float)
    zone_event_count: dict[tuple[int, int], int] = defaultdict(int)

    # Timeline
    minute_buckets: dict[int, dict[str, float]] = defaultdict(
        lambda: {"team_events": 0, "opp_events": 0, "team_xT": 0.0, "opp_xT": 0.0}
    )

    chain_summaries: list[dict] = []

    for chain_indices in possession_chains:
        if not chain_indices:
            continue

        chain_events = [match_events[i] for i in chain_indices]
        chain_team = chain_events[0].get("team", "")

        chain_xT = 0.0
        chain_passes = 0
        chain_duration = 0.0
        reached_final_third = False

        timestamps = [e.get("timestamp", 0.0) for e in chain_events]
        if len(timestamps) > 1:
            chain_duration = max(timestamps) - min(timestamps)
        chain_passes = sum(1 for e in chain_events if e.get("type") == "pass")

        for ev in chain_events:
            etype = ev.get("type", "")
            if etype not in ("pass", "carry"):
                continue
            if not ev.get("completed", True):
                continue

            sx = ev.get("start_x", 0.0)
            sy = ev.get("start_y", 34.0)
            ex = ev.get("end_x", 0.0)
            ey = ev.get("end_y", 34.0)

            xt = xt_model.compute_action_xt(sx, sy, ex, ey)
            chain_xT += xt

            zone = xt_model._zone_from_position(ex, ey)
            zone_event_count[zone] += 1

            if ev.get("team") == team_team:
                gained[zone] += xt
            else:
                conceded[zone] += xt

            # Timeline (per minute)
            minute = int(ev.get("timestamp", 0.0) / 60.0)
            if ev.get("team") == team_team:
                minute_buckets[minute]["team_events"] += 1
                minute_buckets[minute]["team_xT"] += xt
            else:
                minute_buckets[minute]["opp_events"] += 1
                minute_buckets[minute]["opp_xT"] += xt

        # Check if chain reached final third
        for ev in chain_events:
            if ev.get("end_x", 0.0) > 68.0:
                reached_final_third = True
                break

        chain_summaries.append(
            {
                "chain_id": len(chain_summaries),
                "team": chain_team,
                "duration_sec": round(chain_duration, 1),
                "pass_count": chain_passes,
                "xT_gained": round(chain_xT, 4),
                "reached_final_third": reached_final_third,
            }
        )

    # Build cells
    all_zones: set[tuple[int, int]] = set(gained.keys()) | set(conceded.keys())
    cells: list[TerritoryCell] = []
    for zx, zy in all_zones:
        g = gained.get((zx, zy), 0.0)
        c = conceded.get((zx, zy), 0.0)
        cells.append(
            TerritoryCell(
                zone_x=zx,
                zone_y=zy,
                xT_gained=g,
                xT_conceded=c,
                net_xT=g - c,
                possession_time_pct=0.0,
                event_count=zone_event_count.get((zx, zy), 0),
            )
        )

    total_gained = sum(gained.values())
    total_conceded = sum(conceded.values())

    # Dominant zones (>60% net advantage)
    dominant: list[dict] = []
    for cell in cells:
        total = cell.xT_gained + cell.xT_conceded
        if total > 0 and (cell.xT_gained / total) > 0.6:
            dominant.append(
                {
                    "zone_x": cell.zone_x,
                    "zone_y": cell.zone_y,
                    "advantage_pct": round(cell.xT_gained / total * 100, 1),
                    "net_xT": round(cell.net_xT, 4),
                }
            )

    # Territory timeline
    timeline: list[dict] = []
    for minute in sorted(minute_buckets):
        b = minute_buckets[minute]
        total_ev = b["team_events"] + b["opp_events"]
        control_pct = round(b["team_events"] / total_ev * 100, 1) if total_ev else 50.0
        timeline.append(
            {
                "minute": minute,
                "team_control_pct": control_pct,
                "xT_gained_this_min": round(b["team_xT"], 4),
            }
        )

    return TerritoryReport(
        team=team_id,
        match_id=match_id,
        cells=cells,
        total_xT_gained=total_gained,
        total_xT_conceded=total_conceded,
        net_territory_value=total_gained - total_conceded,
        possession_chains=chain_summaries,
        dominant_zones=dominant,
        territory_timeline=timeline,
    )
