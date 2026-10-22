"""xG chain / xG buildup — distributes credit across possession sequences.

xG chain attributes scoring probability to every event in a possession
prior to a shot, while xG buildup specifically credits the first two
passes of a scoring chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kawkab.core.xg_model import compute_xg_from_dict


@dataclass
class XgChain:
    event_idx: int = 0
    event_type: str = ""
    event_team: str = ""
    xg_contribution: float = 0.0
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.event_idx,
            "type": self.event_type,
            "team": self.event_team,
            "xg": round(self.xg_contribution, 4),
            "role": self.role,
        }


@dataclass
class XgBuildup:
    event_idx: int = 0
    event_type: str = ""
    credit: float = 0.0
    is_primary_assist: bool = False
    is_secondary_assist: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.event_idx,
            "type": self.event_type,
            "credit": round(self.credit, 4),
            "primary_assist": self.is_primary_assist,
            "secondary_assist": self.is_secondary_assist,
        }


def compute_xg_chain(
    events: list[dict[str, Any]],
    team: str,
) -> list[XgChain]:
    """Compute xG chain contribution for every event in a team's possessions.

    Each possession sequence ending in a shot has its total xG distributed
    among all events in that sequence, weighted by inverse recency.

    Args:
        events: Chronological list of event dicts with 'type', 'team', 'timestamp'.
        team: Team to analyze ('home' or 'away').

    Returns:
        List of XgChain entries for each event with non-zero contribution.
    """
    results: list[XgChain] = []
    sorted_ev = sorted(
        [e for e in events if e.get("team") == team],
        key=lambda e: e.get("timestamp", 0),
    )

    shot_types = {"shot", "goal"}
    possession_end = {"tackle", "interception", "clearance", "foul", "ball_out"}

    chain_start = 0
    for i, ev in enumerate(sorted_ev):
        if ev.get("type") in possession_end and i > chain_start:
            chain_start = i + 1
            continue
        if ev.get("type") not in shot_types:
            continue

        shot_xg = compute_xg_from_dict(ev)
        if shot_xg <= 0:
            chain_start = i + 1
            continue

        chain_events = sorted_ev[chain_start : i + 1]
        n = len(chain_events)
        for j, cev in enumerate(chain_events):
            position_weight = (j + 1) / n
            contribution = shot_xg * position_weight * 0.5
            role = "shot" if cev.get("type") in shot_types else "buildup"
            results.append(
                XgChain(
                    event_idx=chain_events.index(cev),
                    event_type=cev.get("type", ""),
                    event_team=team,
                    xg_contribution=round(contribution, 4),
                    role=role,
                )
            )

        chain_start = i + 1

    return results


def compute_xg_buildup(
    events: list[dict[str, Any]],
    team: str,
) -> list[XgBuildup]:
    """Compute xG buildup credit for the first two passes before each shot.

    Credits are weighted: primary assist (last pass) gets ~60%,
    secondary assist (second-to-last pass) gets ~40%.

    Args:
        events: Chronological list of event dicts.
        team: Team to analyze.

    Returns:
        List of XgBuildup entries.
    """
    results: list[XgBuildup] = []
    sorted_ev = sorted(
        [e for e in events if e.get("team") == team],
        key=lambda e: e.get("timestamp", 0),
    )

    shot_types = {"shot", "goal"}

    for i, ev in enumerate(sorted_ev):
        if ev.get("type") not in shot_types:
            continue

        shot_xg = compute_xg_from_dict(ev)
        if shot_xg <= 0:
            continue

        preceding = []
        for j in range(i - 1, max(i - 6, -1), -1):
            if sorted_ev[j].get("type") == "pass":
                preceding.append(j)
            elif sorted_ev[j].get("type") in shot_types:
                break
            if len(preceding) == 2:
                break

        for rank, idx in enumerate(preceding):
            is_primary = rank == 0
            credit = shot_xg * (0.6 if is_primary else 0.4)
            results.append(
                XgBuildup(
                    event_idx=idx,
                    event_type="pass",
                    credit=round(credit, 4),
                    is_primary_assist=is_primary,
                    is_secondary_assist=not is_primary,
                )
            )

    return results
