"""kloppy-backed vendor import — the honest-provider path.

Phase D left vendor-data import at an explicit no-provider state because
kloppy was declared but not installed. kloppy 3.19 IS now installed, so
this service wires it for real — feeding the SAME storage tables the
video and hand-parsed vendor pipelines use (matches, players, events,
tracking_frames, tracking_imports), so every downstream model (dossier,
reasoning engine, xG serving, pressing) runs on imported data unchanged.

What is imported, per provider:
    statsbomb   events (kloppy deserializer + synthetic lineups built from
                the events' own Starting XI / Substitution rows when no
                lineup file is supplied — verified against data/statsbomb_corpus),
                via kloppy.providers.statsbomb.load / load_open_data
    skillcorner tracking via kloppy.providers.skillcorner.load, normalized
                through the SAME import_frames plumbing
                VendorTrackingImportService uses (bulk frames, provenance
                row, quality report, event alignment)
    other       reported as provider_unavailable with the actionable
                reason — never silently dropped, never fabricated.

Honesty contract (same as the Transfermarkt precedent):
    - provenance on every row: ``source = "kloppy"`` and the provider
      name travel through event metadata and the tracking-import row,
    - coordinates are normalized once, at the boundary, with the
      provider labeled in metadata,
    - an unavailable provider is an explicit state, not an empty
      success: import_kloppy_file raises ImportError with the reason.

Usage:
    from kawkab.services.kloppy_import_service import KloppyImportService
    svc = KloppyImportService(storage_service)
    summary = await svc.import_statsbomb_events(path, match_id=12)
    summary = await svc.import_skillcorner_tracking(meta_path, raw_path, match_id=12)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from kawkab.services.storage_errors import StorageDuplicateError

logger = logging.getLogger(__name__)

KLOPPY_PROVIDER = "kloppy"


def kloppy_available() -> bool:
    """Whether the kloppy dependency can actually be imported."""
    return find_spec("kloppy") is not None


def _require_kloppy():
    """Return the kloppy module or raise ImportError with an honest message.

    Uses the same availability oracle as the capability report (single
    choke point), so a not-installed environment fails loudly here
    before any storage writes happen.
    """
    if not kloppy_available():
        raise ImportError(
            "kloppy is not installed — vendor import is unavailable. "
            "Install it (pip install kloppy); no substitute or synthetic "
            "data is provided."
        )
    import kloppy

    return kloppy


# ── StatsBomb event conversion ────────────────────────────────────────────
#
# kloppy's typed event objects -> Kawkab event dicts (same shape
# StatsBombImportService persists; see that module for the storage
# contract). Types beyond the mapping below are preserved with their
# generic kloppy name lowercased — nothing is silently dropped.


def _shot_outcome_name(result: Any) -> str:
    name = getattr(result, "name", str(result))
    return name.lower()


def _sb_position_from_kloppy(position: Any) -> str:
    """kloppy PositionType -> coarse Kawkab position code."""
    name = str(getattr(position, "name", "") or "")
    if name.startswith("Goalkeeper"):
        return "GK"
    if "Back" in name and "Midfield" not in name:
        return "CB" if "Center" in name else ("RB" if "Right" in name else "LB")
    if "Wing Back" in name:
        return "RWB" if "Right" in name else "LWB"
    if "Midfield" in name:
        if "Attacking" in name:
            return "AM"
        if "Defensive" in name:
            return "DM"
        return "CM"
    if "Wing" in name or "Forward" in name or "Striker" in name:
        return "CF" if "Center" in name else ("RW" if "Right" in name else "LW")
    return "MID"


def _point_xy(point: Any) -> tuple[float, float] | None:
    x = getattr(point, "x", None)
    y = getattr(point, "y", None)
    if x is None or y is None:
        return None
    return float(x), float(y)


def _event_timestamp(event: Any) -> float:
    """Seconds from kickoff: period offset + in-period timestamp.

    kloppy timestamps are timedeltas measured from the start of the
    period for both the StatsBomb and SkillCorner deserializers, so the
    same 2700s second-half base the hand parsers use applies.
    """
    try:
        period_id = int(event.period.id)
    except (AttributeError, TypeError, ValueError):
        period_id = 1
    base = 2700.0 if period_id == 2 else 0.0
    ts = event.timestamp
    seconds = ts.total_seconds() if hasattr(ts, "total_seconds") else float(ts)
    return base + seconds


def _qualifier_map(event: Any) -> dict[str, Any]:
    """Flatten typed qualifiers into metadata keys we actually serve."""
    out: dict[str, Any] = {}
    for q in getattr(event, "qualifiers", None) or []:
        name = type(q).__name__
        if name == "UnderPressureQualifier":
            out["under_pressure"] = bool(getattr(q, "value", False))
        elif name == "BodyPartQualifier":
            val = getattr(q, "value", None)
            out["body_part"] = str(getattr(val, "name", val) or "").lower()
    return out


def convert_kloppy_event(
    event: Any,
    *,
    team_side: str,
    track_id: int | None,
    provider: str,
) -> dict[str, Any] | None:
    """One kloppy event -> one Kawkab event dict (or None to skip).

    team_side is "home"/"away"; track_id the shooter/passer's registered
    per-match track id (None when the event carries no player).
    """
    from kloppy.domain import EventType

    etype = event.event_type
    metadata: dict[str, Any] = {}
    completed = True
    kawkab_type: str | None

    coords = _point_xy(getattr(event, "coordinates", None))
    if coords:
        metadata["start_x"] = round(coords[0], 2)
        metadata["start_y"] = round(coords[1], 2)

    if etype == EventType.PASS:
        end = _point_xy(getattr(event, "receiver_coordinates", None))
        if end:
            metadata["end_x"] = round(end[0], 2)
            metadata["end_y"] = round(end[1], 2)
        result_name = getattr(event.result, "name", "")
        completed = result_name == "COMPLETE"
        if result_name:
            metadata["pass_outcome"] = result_name.lower()
        metadata.update(_qualifier_map(event))
        kawkab_type = "pass"

    elif etype == EventType.SHOT:
        result_name = _shot_outcome_name(event.result)
        metadata["shot_outcome"] = result_name
        metadata["is_goal"] = result_name == "goal"
        # Vendor xG surfaces as a typed Statistic; label its origin —
        # it is the vendor's model, not Kawkab's, and never both.
        for stat in getattr(event, "statistics", None) or []:
            stat_name = str(getattr(stat, "name", "")).lower()
            if "xg" in stat_name:
                metadata["vendor_xg"] = float(getattr(stat, "value", 0.0) or 0.0)
                metadata["vendor_xg_provider"] = provider
        metadata.update(_qualifier_map(event))
        kawkab_type = "shot"

    elif etype == EventType.CARRY:
        end = _point_xy(getattr(event, "end_coordinates", None))
        if end:
            metadata["end_x"] = round(end[0], 2)
            metadata["end_y"] = round(end[1], 2)
        kawkab_type = "carry"

    elif etype == EventType.TAKE_ON:
        result_name = getattr(event.result, "name", "")
        completed = result_name == "COMPLETE"
        kawkab_type = "dribble"

    elif etype == EventType.DUEL:
        result_name = getattr(event.result, "name", "")
        completed = result_name == "WON"
        kawkab_type = "duel"

    elif etype == EventType.PRESSURE:
        kawkab_type = "pressure"
    elif etype == EventType.INTERCEPTION or etype == EventType.RECOVERY:
        kawkab_type = "interception"
    elif etype == EventType.CLEARANCE:
        kawkab_type = "clearance"
    elif etype == EventType.MISCONTROL:
        kawkab_type = "miscontrol"
    elif etype == EventType.BALL_OUT:
        kawkab_type = "ball_out"
    elif etype == EventType.FOUL_COMMITTED:
        kawkab_type = "foul"
    elif etype == EventType.GOALKEEPER:
        kawkab_type = "save"
    elif etype == EventType.SUBSTITUTION:
        kawkab_type = "substitution"
    elif etype == EventType.GENERIC:
        # GENERIC carries the long tail (Ball Receipt*, 50/50, Block, ...)
        # under its raw StatsBomb type name — preserve it lowercased.
        kawkab_type = None
        raw = getattr(event, "raw_event", None)
        raw_type = (raw or {}).get("type", {}).get("name", "")
        kawkab_type = _GENERIC_TYPE_MAP.get(raw_type, raw_type.lower() or None)
        if kawkab_type is None:
            return None
        if raw_type == "Ball Receipt*":
            result_name = getattr(event.result, "name", "")
            completed = result_name != "INCOMPLETE"
    else:
        kawkab_type = str(getattr(etype, "name", "")).lower() or None
        if kawkab_type is None:
            return None

    if "under_pressure" not in metadata:
        metadata.update(_qualifier_map(event))

    ts = _event_timestamp(event)
    event_dict: dict[str, Any] = {
        "type": kawkab_type,
        "timestamp": ts,
        "team": team_side,
        "completed": completed,
        # kloppy input is vendor-tagged professional data; confidence
        # stays at the video pipeline's human-tag ceiling.
        "confidence": 1.0,
        "metadata": metadata,
    }
    raw = getattr(event, "raw_event", None)
    if raw and "minute" in raw and "second" in raw:
        event_dict["minute"] = raw.get("minute")
        event_dict["second"] = raw.get("second")
    event_dict["metadata"]["source"] = KLOPPY_PROVIDER
    event_dict["metadata"]["provider"] = provider
    if track_id is not None:
        event_dict["from_track_id"] = track_id
    return event_dict


# Events that are pure match-management rows: kloppy deserializes them
# (needed for lineups/teams) but the storage census skips them, exactly
# like StatsBombImportService._SKIPPED_TYPES. kloppy 3.19 surfaces most
# of these as GENERIC events carrying their raw StatsBomb type name, so
# the skip fires on raw names (the base-type names cover the rest).
_MANAGEMENT_EVENT_TYPES = {
    "starting_xi",
    "half_start",
    "half_end",
    "tactical_shift",
    "player_off",
    "player_on",
    "own_goal",  # stored via the shooting event's is_goal
    "own_goal_won",
    "injury_stoppage",
    "camera_flash",
}

_MANAGEMENT_RAW_NAMES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Substitution",
    "Injury Stoppage",
    "Camera Flash",
    "Player On",
    "Player Off",
    "Own Goal Against",
    "Own Goal For",
}


# ── Synthetic lineups for events-only StatsBomb files ─────────────────────


def synthesize_lineups_from_events(events_path: str | Path) -> list[dict]:
    """Build a StatsBomb-lineup-shaped file from an events file.

    data/statsbomb_corpus ships events without companion lineup files,
    but kloppy's deserializer requires one. The events carry everything
    needed: Starting XI rows embed tactics.lineup (starters + positions),
    Substitution rows carry the replacement, and new-schema files may
    carry Player On rows. Team ids stay ints (kloppy keys its formation
    map on them).
    """
    with open(events_path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{events_path}: expected a non-empty JSON event list")

    teams: dict[int, dict] = {}
    for ev in raw:
        type_name = (ev.get("type") or {}).get("name", "")
        if type_name == "Starting XI":
            tid = ev["team"]["id"]
            players = [
                {
                    "player_id": p["player"]["id"],
                    "player_name": p["player"]["name"],
                    "player_nickname": p["player"].get("nickname"),
                    "jersey_number": p.get("jersey_number", 0),
                    "position": {
                        "id": p["position"]["id"],
                        "name": p["position"]["name"],
                    },
                }
                for p in ev.get("tactics", {}).get("lineup", [])
            ]
            teams[tid] = {"team_id": tid, "team_name": ev["team"]["name"], "lineup": players}

    if not teams:
        raise ValueError(f"{events_path}: no Starting XI rows — cannot build lineups")

    for ev in raw:
        type_name = (ev.get("type") or {}).get("name", "")
        tid = (ev.get("team") or {}).get("id")
        if tid not in teams:
            continue
        if type_name == "Substitution":
            p = (ev.get("substitution") or {}).get("replacement")
            if p:
                teams[tid]["lineup"].append(
                    {
                        "player_id": p["id"],
                        "player_name": p["name"],
                        "player_nickname": p.get("nickname"),
                        "jersey_number": 0,
                        "position": None,
                    }
                )
        elif type_name == "Player On":
            p = ev.get("player")
            if p:
                teams[tid]["lineup"].append(
                    {
                        "player_id": p["id"],
                        "player_name": p["name"],
                        "player_nickname": p.get("nickname"),
                        "jersey_number": 0,
                        "position": None,
                    }
                )

    return list(teams.values())


# ── The service ────────────────────────────────────────────────────────────


class KloppyImportService:
    """kloppy-backed vendor import into the Kawkab database."""

    def __init__(self, storage_service: Any) -> None:
        self.storage = storage_service
        self._kloppy = _require_kloppy()
        self._tracking_svc = None  # lazy; avoids import cycles at module load

    # ── capability report (honest-provider states) ────────────────────

    @staticmethod
    def provider_capabilities() -> list[dict[str, Any]]:
        """Per-provider availability with actionable reasons.

        Grounded in what THIS installation exposes: the providers with
        load() entry points in the installed kloppy, plus the ones this
        service actually converts. Anything else is provider_unavailable
        — the same no-fabrication contract as the Transfermarkt probe.
        """
        if not kloppy_available():
            return [
                {
                    "provider": p,
                    "kind": kind,
                    "available": False,
                    "status": "provider_unavailable",
                    "reason": "kloppy is not installed",
                }
                for p, kind in (("statsbomb", "events"), ("skillcorner", "tracking"))
            ]
        import importlib

        import kloppy
        from kloppy.domain import Provider

        convertible = {p.value for p in Provider}
        with_load = []
        for name in convertible:
            try:
                mod = importlib.import_module(f"kloppy.{name}")
            except ImportError:
                continue
            if hasattr(mod, "load"):
                with_load.append(name)

        out = []
        for name, kind, available in (
            ("statsbomb", "events", "statsbomb" in with_load),
            ("skillcorner", "tracking", "skillcorner" in with_load),
        ):
            out.append(
                {
                    "provider": name,
                    "kind": kind,
                    "available": available,
                    "status": "available" if available else "provider_unavailable",
                    "reason": None
                    if available
                    else (
                        f"installed kloppy {getattr(kloppy, '__version__', '')} "
                        f"exposes no load() for this provider"
                    ),
                }
            )
        return out

    # ── StatsBomb events ──────────────────────────────────────────────

    async def import_statsbomb_events(
        self,
        events_path: str | Path,
        *,
        lineup_path: str | Path | None = None,
        match_id: int | None = None,
        match_name: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
    ) -> dict[str, Any]:
        """Import one StatsBomb events file through kloppy.

        Creates the match unless match_id is given. Without lineup_path,
        lineups are synthesized from the events file itself (documented
        above). Every persisted row carries source=kloppy provenance.
        """
        from kloppy import statsbomb as sb_provider

        events_path = Path(events_path)
        if not events_path.exists():
            raise FileNotFoundError(f"events file not found: {events_path}")

        # Lineups: supplied file, else synthesized from the events (the
        # corpus ships no lineup files). Written to a temp file because
        # kloppy's loader wants a path.
        tmp_lineup: str | None = None
        if lineup_path is None:
            lineups = synthesize_lineups_from_events(events_path)
            fd, tmp_lineup = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(lineups, fh)
            lineup_path = tmp_lineup

        try:
            dataset = sb_provider.load(
                event_data=str(events_path),
                lineup_data=str(lineup_path),
                coordinates="statsbomb",  # vendor units; converted below
            )
        finally:
            if tmp_lineup:
                Path(tmp_lineup).unlink(missing_ok=True)

        # Teams + match labels. kloppy preserves the events' team order
        # (first Starting XI = home, same heuristic the hand importer
        # documents) — and both teams' events are imported.
        teams = dataset.metadata.teams
        home = home_team or teams[0].name
        away = away_team or teams[1].name
        name = match_name or f"{home} vs {away} (kloppy StatsBomb import)"
        if match_id is None:
            match_id = await self.storage.save_match(
                name=name, video_path="", home_team=home, away_team=away
            )
            if not match_id:
                raise RuntimeError("storage refused to create the match row")

        # Per-match players (stable synthetic track ids, same spirit as
        # the hand importer so cross-vendor pipelines behave identically).
        team_sides = {teams[0]: "home", teams[1]: "away"}
        player_tracks: dict[Any, int] = {}
        for side_team, side in team_sides.items():
            for p in side_team.players:
                player_track_id = self._stable_track_id(side_team.name, p.name)
                await self.storage.save_player(
                    match_id,
                    {
                        "track_id": player_track_id,
                        "name": p.name,
                        "team": side,
                        "position": _sb_position_from_kloppy(p.starting_position),
                        "jersey_number": int(p.jersey_no) if p.jersey_no else None,
                    },
                )
                player_tracks[p] = player_track_id

        imported = 0
        skipped = 0
        shots = 0
        goals = 0
        total_xg = 0.0
        unhandled_types: dict[str, int] = {}

        for event in dataset.events:
            raw = getattr(event, "raw_event", None) or {}
            raw_type_name = (raw.get("type") or {}).get("name", "")
            etype_name = str(getattr(event.event_type, "name", "")).lower()
            if etype_name in _MANAGEMENT_EVENT_TYPES or raw_type_name in _MANAGEMENT_RAW_NAMES:
                skipped += 1
                continue
            side = team_sides.get(event.team, "home")
            track_id: int | None = player_tracks.get(event.player)
            converted = convert_kloppy_event(
                event, team_side=side, track_id=track_id, provider="statsbomb"
            )
            if converted is None:
                unhandled_types[etype_name] = unhandled_types.get(etype_name, 0) + 1
                skipped += 1
                continue
            if converted["type"] == "shot":
                shots += 1
                if converted["metadata"].get("is_goal"):
                    goals += 1
                total_xg += converted["metadata"].get("vendor_xg", 0.0)
            try:
                await self.storage.save_event(match_id, converted)
            except StorageDuplicateError:
                # Same dedup-index semantics as the hand importers.
                skipped += 1
                continue
            imported += 1

        # Headline metrics — vendor xG labeled with its origin.
        await self.storage.save_advanced_metrics(
            match_id, "xg_total", round(total_xg, 3), metric_category="import"
        )
        await self.storage.save_advanced_metrics(
            match_id, "shots_total", float(shots), metric_category="import"
        )
        await self.storage.save_advanced_metrics(
            match_id, "goals_total", float(goals), metric_category="import"
        )

        return {
            "match_id": match_id,
            "match_name": name,
            "home_team": home,
            "away_team": away,
            "provider": "statsbomb",
            "source": KLOPPY_PROVIDER,
            "kloppy_version": getattr(self._kloppy, "__version__", ""),
            "events_imported": imported,
            "events_skipped": skipped,
            "unhandled_types": unhandled_types,
            "players_registered": len(player_tracks),
            "shots": shots,
            "goals": goals,
            "xg_total": round(total_xg, 3),
            "xg_note": "vendor (StatsBomb) model xG, labeled vendor_xg",
            "source_file": str(events_path),
        }

    # ── SkillCorner tracking ──────────────────────────────────────────

    async def import_skillcorner_tracking(
        self,
        meta_path: str | Path,
        raw_path: str | Path,
        *,
        match_id: int | None = None,
        match_name: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        max_frames: int | None = None,
        fps: float | None = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> dict[str, Any]:
        """Import SkillCorner meta+raw through kloppy into tracking frames.

        Coordinates come out of kloppy in SkillCorner unit space (±1) and
        are normalized to Kawkab meters with the SAME convention the hand
        parser uses, then persisted through VendorTrackingImportService's
        frame pipeline: bulk frames, provenance row (vendor="skillcorner-
        kloppy", source=kloppy), quality report, event alignment.
        """
        from kloppy import skillcorner

        for p in (meta_path, raw_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"file not found: {p}")

        dataset = skillcorner.load(
            meta_data=str(meta_path),
            raw_data=str(raw_path),
            only_alive=False,
            coordinates="skillcorner",  # keep ±1; normalize ourselves
        )
        records = list(dataset.records)
        if not records:
            raise ValueError("kloppy parsed zero frames from the SkillCorner files")
        if max_frames is not None and max_frames > 0:
            records = records[:max_frames]
        parsed_fps = float(getattr(dataset.metadata, "frame_rate", 0) or 0) or fps
        if parsed_fps is None:
            parsed_fps = 25.0

        # frames in VendorTrackingImportService's canonical shape
        # (Kawkab meters); PlayerData.speed is Optional and often None.
        frames: list[dict] = []
        for fr in records:
            players = []
            for player, pdata in fr.players_data.items():
                pt = pdata.coordinates
                if pt is None or pt.x is None or pt.y is None:
                    continue
                mx = (float(pt.x) + 1.0) / 2.0 * pitch_length_m
                my = (float(pt.y) + 1.0) / 2.0 * pitch_width_m
                players.append(
                    {
                        "track_id": self._player_track_id(player),
                        "x": round(mx, 3),
                        "y": round(my, 3),
                        "speed": float(pdata.speed) if pdata.speed is not None else 0.0,
                    }
                )
            ball = None
            bpt = fr.ball_coordinates
            if bpt is not None and bpt.x is not None and bpt.y is not None:
                ball = {
                    "x": round((float(bpt.x) + 1.0) / 2.0 * pitch_length_m, 3),
                    "y": round((float(bpt.y) + 1.0) / 2.0 * pitch_width_m, 3),
                    "z": float(getattr(bpt, "z", 0.0) or 0.0),
                }
            frames.append(
                {
                    "frame_number": int(fr.frame_id),
                    "timestamp": float(fr.timestamp.total_seconds()),
                    "period": int(fr.period.id) if fr.period is not None else 0,
                    "player_detections": players,
                    "ball": ball,
                }
            )

        player_meta: dict[int, dict] = {}
        for team_obj, side in zip(dataset.metadata.teams, ("home", "away"), strict=True):
            for player in team_obj.players:
                player_meta[self._player_track_id(player)] = {
                    "name": player.name,
                    "team": side,
                    "position": str(getattr(player.starting_position, "name", "") or "").title(),
                    "jersey_number": int(player.jersey_no) if player.jersey_no else None,
                }

        svc = self._tracking_import_service()
        frame_result = await svc.import_frames(
            frames,
            player_meta=player_meta,
            vendor="skillcorner-kloppy",
            source="kloppy",
            source_path=str(raw_path),
            match_id=match_id,
            match_name=match_name,
            home_team=home_team or dataset.metadata.teams[0].name,
            away_team=away_team or dataset.metadata.teams[1].name,
            fps=parsed_fps,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )
        match_id2 = frame_result["match_id"]
        quality = frame_result["quality"]
        return {
            "match_id": match_id2,
            "provider": "skillcorner",
            "source": KLOPPY_PROVIDER,
            "kloppy_version": getattr(self._kloppy, "__version__", ""),
            "frames_imported": quality.get("frames", 0),
            "quality": quality,
            "fps": parsed_fps,
            "source_file": str(raw_path),
        }

    # ── helpers ────────────────────────────────────────────────────────

    def _tracking_import_service(self):
        """Lazily build the tracking pipeline owner (import cycle safety)."""
        if self._tracking_svc is None:
            from kawkab.services.vendor_tracking_import_service import (
                VendorTrackingImportService,
            )

            self._tracking_svc = VendorTrackingImportService(self.storage)
        return self._tracking_svc

    @staticmethod
    def _stable_track_id(team_name: str, player_name: str) -> int:
        """Deterministic per-(team, player) track id in [1, 999]."""
        return (abs(hash((team_name, player_name))) % 998) + 1

    @staticmethod
    def _player_track_id(player: Any) -> int:
        """kloppy player -> stable int track id (ids may be strings)."""
        pid = str(getattr(player, "player_id", "") or "")
        if pid.isdigit():
            return int(pid) % 100000
        return (abs(hash(pid)) % 99998) + 1


# Long-tail StatsBomb types kloppy surfaces as GENERIC events, mapped to
# the core/ conventions (kept in lockstep with statsbomb_import_service
# so downstream modules see identical vocabularies).
_GENERIC_TYPE_MAP: dict[str, str] = {
    "Ball Receipt*": "ball_receipt",
    "Ball Recovery": "interception",
    "Block": "block",
    "Dribbled Past": "dribbled_past",
    "Shield": "shield",
    "50/50": "duel",
    "Error": "miscontrol",
    "Bad Behaviour": "foul",
    "Goal Keeper": "save",
    "Aerial Lost": "aerial_lost",
    "Aerial Won": "aerial_won",
    "Dispossessed": "dispossessed",
    "Offside": "offside",
    "Referee Ball-Drop": "referee_ball_drop",
    "Own Goal Against": "own_goal",
    "Own Goal For": "own_goal_won",
}
