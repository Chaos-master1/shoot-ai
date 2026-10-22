"""Storage rows -> reasoning-engine inputs: one owner, one shape.

The reasoning engine (``ReasoningService`` and the pattern checkers) is
written against a *normalized* event shape: ``type`` instead of
``event_type``, and the zone/situation/outcome/flank fields that the
storage layer keeps inside a JSON ``metadata`` string merged onto the
row. It also consumes a ``MatchAnalysis`` built from those events.

Both the plan generator and the intervention-efficacy service must feed
the engine the same shape, otherwise they measure different things from
the same match. Before this module existed, the normalization lived as
private static methods on ``AnalysisHandler`` -- so any service outside
the UI layer that wanted to reason over stored events had to either
duplicate it (and drift) or skip it (and read empty buckets). These
functions are that boundary, in one place.

Pure functions: no I/O, no logging, no service state.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kawkab.services.analysis_service import MatchAnalysis


def normalize_storage_events(events: Any) -> list[dict]:
    """Storage rows -> the event shape ``ReasoningService`` expects.

    - ``event_type`` becomes ``type`` (kept if already present)
    - the JSON ``metadata`` string is parsed and merged *without*
      overwriting explicit columns (``setdefault``): a real column always
      wins over a metadata echo of itself
    - rows that are not dicts are skipped rather than raising

    Returns a new list of new dicts; the caller's rows are not mutated.
    """
    normalized: list[dict] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        item = dict(ev)
        if "type" not in item:
            item["type"] = item.get("event_type", "")
        meta = item.get("metadata")
        if isinstance(meta, str) and meta:
            try:
                parsed = json.loads(meta)
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        item.setdefault(key, value)
            except json.JSONDecodeError:
                pass
        normalized.append(item)
    return normalized


def build_match_analysis(
    match_id: int, match_data: dict, normalized_events: list[dict]
) -> MatchAnalysis:
    """Build the ``MatchAnalysis`` the reasoning engine consumes.

    Team totals are derived from the *normalized* events only. Nothing
    is invented: a match with no events yields zero totals, and the
    engine returns no diagnoses for it.
    """
    from kawkab.services.analysis_service import MatchAnalysis, TeamStats

    home = TeamStats(team_name=str((match_data or {}).get("home_team", "Home")))
    away = TeamStats(team_name=str((match_data or {}).get("away_team", "Away")))
    away_name = str((match_data or {}).get("away_team", ""))

    for ev in normalized_events:
        etype = str(ev.get("type", ""))
        team = str(ev.get("team", "home"))
        target = away if team == "away" or team == away_name else home
        if etype == "pass":
            target.passes_attempted += 1
            if ev.get("completed"):
                target.passes_completed += 1
        elif etype == "shot":
            target.shots += 1
        elif etype == "tackle":
            target.tackles += 1
        elif etype == "corner":
            target.corners += 1
        elif etype == "foul":
            target.fouls += 1

    duration = float((match_data or {}).get("duration") or 0.0)
    if duration <= 0 and normalized_events:
        duration = max(
            (float(e.get("timestamp", 0) or 0) for e in normalized_events),
            default=0.0,
        )

    return MatchAnalysis(
        match_id=match_id,
        duration_seconds=duration,
        home_team=home,
        away_team=away,
        players={},
        events=normalized_events,
    )
