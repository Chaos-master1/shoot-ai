"""Data reconciliation service — merge events from multiple sources (tracking, Opta, wearable, provider)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

SOURCE_PRIORITY = {"provider": 0, "tracking": 1, "events": 2, "wearable": 3}

EVENT_TYPE_KEYS = {"type", "event_type", "event"}
TEAM_KEYS = {"team", "team_id", "team_name"}
TIME_KEYS = {"timestamp", "time", "ts", "minute"}
X_KEYS = {"x", "start_x", "coord_x"}
Y_KEYS = {"y", "start_y", "coord_y"}


@dataclass
class DataSource:
    source_id: str
    source_type: str
    data: list[dict] = field(default_factory=list)


@dataclass
class ReconciliationMatch:
    match_id: str
    home_team: str
    away_team: str
    sources: list[DataSource] = field(default_factory=list)
    reconciled_events: list[dict] | None = None


class DataReconciliationService:
    def __init__(self):
        self._matches: dict[str, ReconciliationMatch] = {}

    def add_source(
        self,
        match_id: str,
        source_id: str,
        source_type: str,
        data: list[dict],
    ) -> None:
        if source_type not in SOURCE_PRIORITY:
            logger.warning(
                f"Unknown source_type '{source_type}' for source '{source_id}' — will be treated lowest priority"
            )

        if match_id not in self._matches:
            self._matches[match_id] = ReconciliationMatch(
                match_id=match_id,
                home_team="",
                away_team="",
            )

        existing = [s for s in self._matches[match_id].sources if s.source_id == source_id]
        if existing:
            existing[0].data = data
            existing[0].source_type = source_type
            logger.info(f"Updated source '{source_id}' ({source_type}) for match '{match_id}'")
        else:
            self._matches[match_id].sources.append(
                DataSource(source_id=source_id, source_type=source_type, data=data),
            )
            logger.info(f"Added source '{source_id}' ({source_type}) to match '{match_id}'")

    def remove_source(self, match_id: str, source_id: str) -> None:
        match = self._matches.get(match_id)
        if not match:
            logger.warning(f"Cannot remove source '{source_id}' — match '{match_id}' not found")
            return
        match.sources = [s for s in match.sources if s.source_id != source_id]
        logger.info(f"Removed source '{source_id}' from match '{match_id}'")

    def reconcile(self, match_id: str, time_tolerance_ms: int = 2000) -> list[dict]:
        match = self._matches.get(match_id)
        if not match or not match.sources:
            return []

        all_events: list[dict] = []
        for src in match.sources:
            for ev in src.data:
                all_events.append(
                    {**ev, "_source_id": src.source_id, "_source_type": src.source_type}
                )

        time_tolerance_s = time_tolerance_ms / 1000.0
        reconciled: list[dict] = []

        while all_events:
            anchor = all_events.pop(0)
            anchor_ts = self._extract_time(anchor)
            anchor_type = self._extract_type(anchor)
            anchor_team = self._extract_team(anchor)
            merged = copy.deepcopy(anchor)
            merged["_sources"] = {anchor["_source_id"]}

            i = 0
            while i < len(all_events):
                other = all_events[i]
                other_ts = self._extract_time(other)
                if (
                    other_ts is not None
                    and anchor_ts is not None
                    and abs(other_ts - anchor_ts) > time_tolerance_s
                ):
                    i += 1
                    continue

                other_type = self._extract_type(other)
                types_match = (
                    other_type is not None and anchor_type is not None and other_type == anchor_type
                )

                other_team = self._extract_team(other)
                teams_match = (
                    other_team is not None and anchor_team is not None and other_team == anchor_team
                )

                if types_match and teams_match:
                    self._merge_events(merged, other)
                    merged["_sources"].add(other["_source_id"])
                    all_events.pop(i)
                else:
                    i += 1

            reconciled.append(merged)

        match.reconciled_events = reconciled
        logger.info(f"Reconciled {len(reconciled)} events for match '{match_id}'")
        return reconciled

    def get_coverage(self, match_id: str) -> dict[str, dict]:
        match = self._matches.get(match_id)
        if not match or not match.sources:
            return {}

        total = len(match.sources[0].data) if match.sources else 0
        max_count = max((len(s.data) for s in match.sources), default=0)
        denominator = max(total, max_count)
        if denominator == 0:
            return {
                s.source_id: {"event_count": 0, "coverage_pct": 0.0, "source_type": s.source_type}
                for s in match.sources
            }

        coverage = {}
        for src in match.sources:
            pct = round(len(src.data) / denominator * 100, 1)
            coverage[src.source_id] = {
                "event_count": len(src.data),
                "coverage_pct": pct,
                "source_type": src.source_type,
            }
        return coverage

    def get_conflicts(self, match_id: str) -> list[dict]:
        match = self._matches.get(match_id)
        if not match or len(match.sources) < 2:
            return []

        all_events: list[dict] = []
        for src in match.sources:
            for ev in src.data:
                all_events.append(
                    {**ev, "_source_id": src.source_id, "_source_type": src.source_type}
                )

        time_tolerance_s = 2.0
        conflicts = []

        for i in range(len(all_events)):
            for j in range(i + 1, len(all_events)):
                a, b = all_events[i], all_events[j]
                if a["_source_id"] == b["_source_id"]:
                    continue
                a_ts = self._extract_time(a)
                b_ts = self._extract_time(b)
                if a_ts is None or b_ts is None:
                    continue
                if abs(a_ts - b_ts) > time_tolerance_s:
                    continue

                a_type = self._extract_type(a)
                b_type = self._extract_type(b)
                a_team = self._extract_team(a)
                b_team = self._extract_team(b)

                type_conflict = a_type is not None and b_type is not None and a_type != b_type
                team_conflict = a_team is not None and b_team is not None and a_team != b_team

                if type_conflict or team_conflict:
                    conflicts.append(
                        {
                            "anchor_event": a,
                            "conflicting_event": b,
                            "type_conflict": type_conflict,
                            "team_conflict": team_conflict,
                            "time_delta_ms": round(abs(a_ts - b_ts) * 1000, 1),
                        }
                    )

        return conflicts

    def clear(self, match_id: str) -> None:
        if match_id in self._matches:
            del self._matches[match_id]
            logger.info(f"Cleared all data for match '{match_id}'")

    @staticmethod
    def _extract_time(event: dict) -> float | None:
        for key in TIME_KEYS:
            val = event.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def _extract_type(event: dict) -> str | None:
        for key in EVENT_TYPE_KEYS:
            val = event.get(key)
            if val is not None:
                return str(val)
        return None

    @staticmethod
    def _extract_team(event: dict) -> str | None:
        for key in TEAM_KEYS:
            val = event.get(key)
            if val is not None:
                return str(val)
        return None

    @staticmethod
    def _extract_x(event: dict) -> float | None:
        for key in X_KEYS:
            val = event.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def _extract_y(event: dict) -> float | None:
        for key in Y_KEYS:
            val = event.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def _merge_events(target: dict, source: dict) -> None:
        target_type = target.get("_source_type", "wearable")
        source_type = source.get("_source_type", "wearable")
        target_prio = SOURCE_PRIORITY.get(target_type, 99)
        source_prio = SOURCE_PRIORITY.get(source_type, 99)

        for k, v in source.items():
            if k.startswith("_"):
                continue
            if k not in target or target[k] is None:
                target[k] = v
            else:
                needs_coord = any(prefix in k.lower() for prefix in ["x", "y", "coord", "start_"])
                if needs_coord:
                    continue
                if source_prio < target_prio:
                    target[k] = v

        target_x = DataReconciliationService._extract_x(target)
        target_y = DataReconciliationService._extract_y(target)
        source_x = DataReconciliationService._extract_x(source)
        source_y = DataReconciliationService._extract_y(source)

        if source_x is not None and (target_x is None or source_prio < target_prio):
            for key in X_KEYS:
                if key in source:
                    target[key] = source[key]
                    break

        if source_y is not None and (target_y is None or source_prio < target_prio):
            for key in Y_KEYS:
                if key in source:
                    target[key] = source[key]
                    break
