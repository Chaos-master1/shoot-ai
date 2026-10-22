"""Vendor event-data import — Opta F24 / Wyscout into the Kawkab database.

Rescues the two orphaned parsers (``opta_importer.OptaF7Importer``,
``wyscout_importer.WyscoutImporter``) into the same storage-backed import
path StatsBomb already has: one service, per-vendor converters, match +
players + events persisted through StorageService so every downstream
core/ model runs on vendor event data with zero video capture.

Design notes:
    - Event types: the Opta parser already maps its numeric type ids to
      snake_case names (``_opta_type_name``); this service maps the shot
      family onto Kawkab's single "shot" type and passes everything else
      through (never dropped) — same invariant as statsbomb_import_service.
    - Coordinates: both vendors use a 0-100 x 0-100 grid; normalized to
      Kawkab meters (105x68) here.
    - Timestamps: both vendors' standard exports use continuous match
      minutes (46+ = second half); stored as minute*60+second as parsed.
    - Identity: Opta events carry team/player uIDs, not names. When an F7
      file is supplied, uIDs are resolved to display names; otherwise a
      first-seen/second-seen team heuristic (documented, same spirit as
      StatsBomb's) plus "Opta <uID>" player labels are used.
    - xG: Opta and Wyscout feeds ship no xG; the field is cached as 0.0
      and Kawkab's own trained model can be applied downstream.

Usage:
    from kawkab.services.vendor_event_import_service import (
        VendorEventImportService,
    )
    svc = VendorEventImportService(storage)
    summary = await svc.import_opta_f24(f24_path, f7_path=f7_path)
    summary = await svc.import_wyscout(path)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KAWKAB_X_MAX = 105.0
KAWKAB_Y_MAX = 68.0

# Parser-mapped Opta event names in the shot family -> Kawkab "shot"
_OPTA_SHOT_NAMES = {"shot_off_target", "shot_on_target", "shot_post", "goal", "miss"}

_OPTA_SHOT_OUTCOME = {
    "goal": "goal",
    "shot_on_target": "saved",
    "shot_post": "post",
    "shot_off_target": "off_target",
    "miss": "off_target",
}

# Wyscout event names -> Kawkab event types (subset; others preserved)
_WYSCOUT_TYPE_MAP: dict[str, str] = {
    "Passes": "pass",
    "Shots": "shot",
    "Duels": "duel",
    "Interruptions": "interception",
    "Clearances": "clearance",
    "Fouls": "foul",
}


def _norm100_to_meters(x: float, y: float) -> tuple[float, float]:
    """Opta/Wyscout 0-100 grid -> Kawkab meters."""
    return x / 100.0 * KAWKAB_X_MAX, y / 100.0 * KAWKAB_Y_MAX


def parse_f7_identity(f7_xml: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Extract identity maps from an F7 file.

    Returns (teams, players, sides):
        teams   {team_uID: display name}
        players {player_uID: display name}
        sides   {"home": team_uID, "away": team_uID}  (either may be
                missing when the file doesn't carry TeamData Side attrs)

    The existing OptaF7Importer.parse_lineup_xml returns player_refs
    without display names, which is exactly what event-side resolution
    needs — so this does the minimal extra walk here rather than
    modifying a tested parser.

    Real Opta F7 convention: ``<TeamData Side="Home" TeamRef="t123">``
    under MatchData carries the home/away assignment; ``<Team uID="t123">
    <Name>...</Name></Team>`` carries the display names. The TeamData
    Side attribute is checked case-insensitively because exporters vary.
    """
    teams: dict[str, str] = {}
    players: dict[str, str] = {}
    sides: dict[str, str] = {}
    try:
        root = ET.fromstring(f7_xml)
    except ET.ParseError as exc:
        logger.warning("F7 identity parse failed: %s", exc)
        return teams, players, sides

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    for el in root.iter():
        local = _local(el.tag)
        if local == "TeamData":
            attrs_ci = {k.lower(): v for k, v in el.attrib.items()}
            side = attrs_ci.get("side", "").lower()
            team_ref = attrs_ci.get("teamref") or attrs_ci.get("ref") or ""
            team_ref = team_ref.lstrip("t")
            if side in ("home", "away") and team_ref:
                sides[side] = team_ref
        elif local == "Team":
            uid = el.get("uID", "").lstrip("t")
            if not uid:
                continue
            for child in el.iter():
                if _local(child.tag) == "Name" and child.text:
                    teams[uid] = child.text.strip()
                    break
            for player_el in el.iter():
                if _local(player_el.tag) != "Player":
                    continue
                p_uid = player_el.get("uID", "").lstrip("p")
                for child in player_el.iter():
                    if _local(child.tag) == "Name" and child.text:
                        players[p_uid] = child.text.strip()
                        break
    return teams, players, sides


class VendorEventImportService:
    """Imports Opta F24 / Wyscout event files as first-class Kawkab matches."""

    def __init__(self, storage_service: Any) -> None:
        self.storage = storage_service

    # ── Opta ────────────────────────────────────────────────────────────

    async def import_opta_f24(
        self,
        f24_path: str | Path,
        f7_path: str | Path | None = None,
        *,
        match_name: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        match_id: int | None = None,
    ) -> dict[str, Any]:
        """Import an Opta F24 events XML (optionally paired with F7 match info)."""
        from kawkab.services.opta_importer import OptaF7Importer

        importer = OptaF7Importer()

        team_names: dict[str, str] = {}
        player_names: dict[str, str] = {}
        f7_sides: dict[str, str] = {}
        f7_home = f7_away = ""
        if f7_path is not None and Path(f7_path).exists():
            f7_xml = Path(f7_path).read_text(encoding="utf-8")
            team_names, player_names, f7_sides = parse_f7_identity(f7_xml)
            if f7_sides.get("home"):
                f7_home = team_names.get(f7_sides["home"], "")
            if f7_sides.get("away"):
                f7_away = team_names.get(f7_sides["away"], "")
            elif team_names:
                ordered = list(team_names.items())
                f7_home = ordered[0][1]
                f7_away = ordered[1][1] if len(ordered) > 1 else ""

        home = home_team or f7_home or "Home"
        away = away_team or f7_away or "Away"
        name = match_name or f"{home} vs {away} (Opta import)"

        if match_id is None:
            match_id = await self.storage.save_match(
                name=name, video_path="", home_team=home, away_team=away
            )
            if not match_id:
                raise RuntimeError("storage refused to create the match row")

        raw_events = importer.parse_event_xml(Path(f24_path).read_text(encoding="utf-8"))
        if not raw_events:
            raise ValueError(f"{f24_path}: parsed zero Opta events")

        # Team uID -> home/away. Priority: (1) F7 TeamData Side attrs,
        # (2) F7 Team document order (first=home — Opta lists the home
        # side first in real exports), (3) first-seen team in the event
        # stream = home (documented heuristic, same spirit as StatsBomb's).
        uid_to_side: dict[str, str] = {}
        if f7_sides.get("home"):
            uid_to_side[f7_sides["home"]] = "home"
        if f7_sides.get("away"):
            uid_to_side[f7_sides["away"]] = "away"
        if not uid_to_side and team_names:
            for i, uid in enumerate(team_names):
                uid_to_side[uid] = "home" if i == 0 else "away"
        for ev in raw_events:
            uid = str(ev.team or "").lstrip("t")
            if uid and uid not in uid_to_side:
                uid_to_side[uid] = "home" if "home" not in uid_to_side.values() else "away"

        imported = 0
        skipped = 0
        players_registered = 0
        shots = 0
        goals = 0
        total_xg = 0.0
        seen: set[str] = set()
        track_ids: dict[str, int] = {}

        for ev in raw_events:
            uid = str(ev.team or "").lstrip("t")
            team = uid_to_side.get(uid, "home")
            player_uid = str(ev.player or "").lstrip("p")
            player_name = player_names.get(player_uid) or (
                f"Opta {player_uid}" if player_uid else ""
            )

            if player_uid and player_uid not in seen:
                seen.add(player_uid)
                tid = self._stable_track_id(team, player_name or player_uid)
                await self.storage.save_player(
                    match_id,
                    {
                        "track_id": tid,
                        "name": player_name or f"Player {player_uid}",
                        "team": team,
                        "position": None,
                        "jersey_number": None,
                    },
                )
                track_ids[player_uid] = tid
                players_registered += 1

            kawkab_ev = self._opta_to_kawkab_event(ev, team, track_ids.get(player_uid))
            if kawkab_ev is None:
                skipped += 1
                continue
            if kawkab_ev["type"] == "shot":
                shots += 1
                meta = kawkab_ev.get("metadata", {})
                if meta.get("is_goal"):
                    goals += 1
                total_xg += meta.get("xg", 0.0)
            saved = await self.storage.save_event(match_id, kawkab_ev)
            if saved:
                imported += 1
            else:
                skipped += 1

        await self._cache_totals(match_id, imported, shots, goals, total_xg)
        return {
            "match_id": match_id,
            "match_name": name,
            "vendor": "opta",
            "events_imported": imported,
            "events_skipped": skipped,
            "players_registered": players_registered,
            "shots": shots,
            "goals": goals,
            "xg_total": round(total_xg, 3),
            "source_file": str(f24_path),
        }

    def _opta_to_kawkab_event(
        self,
        ev: Any,
        team: str,
        from_track: int | None,
    ) -> dict[str, Any] | None:
        """One Opta ProviderEvent -> one Kawkab event dict (or None)."""
        parser_type = (ev.type or "").lower()
        type_id = str(ev.extra.get("type_id", "") or "")

        ktype: str | None
        if parser_type in _OPTA_SHOT_NAMES:
            ktype = "shot"
        elif parser_type:
            ktype = parser_type
        else:
            ktype = f"opta_{type_id}" if type_id else None
        if ktype is None:
            return None

        sx, sy = _norm100_to_meters(ev.x, ev.y)
        ex, ey = _norm100_to_meters(ev.end_x, ev.end_y)
        meta: dict[str, Any] = {
            "start_x": round(sx, 2),
            "start_y": round(sy, 2),
            "end_x": round(ex, 2),
            "end_y": round(ey, 2),
            "opta_event_id": ev.event_id,
            "opta_type_id": type_id,
            "source": "opta",
        }
        if ktype == "shot":
            meta["is_goal"] = parser_type == "goal"
            meta["shot_outcome"] = _OPTA_SHOT_OUTCOME.get(parser_type, "off_target")
            meta["xg"] = 0.0  # Opta feeds ship no xG; trained model applies downstream
        quals = ev.extra.get("qualifiers")
        if quals:
            meta["opta_quals"] = quals

        return {
            "type": ktype,
            "timestamp": float(ev.timestamp or 0.0),
            "team": team,
            "completed": bool(ev.outcome),
            "from_track_id": from_track,
            "metadata": meta,
        }

    # ── Wyscout ─────────────────────────────────────────────────────────

    async def import_wyscout(
        self,
        path: str | Path,
        *,
        match_name: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        match_id: int | None = None,
    ) -> dict[str, Any]:
        """Import a Wyscout JSON export (local file mode, no API key)."""
        from kawkab.services.wyscout_importer import WyscoutImporter

        importer = WyscoutImporter()
        match, events, lineups = importer.import_local(path)
        if match is None and not events:
            raise ValueError(f"{path}: parsed zero Wyscout events (bad file?)")

        home = home_team or (match.home_team if match else "") or "Home"
        away = away_team or (match.away_team if match else "") or "Away"
        name = match_name or (f"{home} vs {away} (Wyscout import)")

        if match_id is None:
            match_id = await self.storage.save_match(
                name=name, video_path="", home_team=home, away_team=away
            )
            if not match_id:
                raise RuntimeError("storage refused to create the match row")

        # Lineups: names + shirt numbers when available
        lineup_players: dict[str, dict] = {}
        for lu in lineups:
            team_of_lineup = "home" if lu.get("team_name") == home else "away"
            for p in lu.get("players", []):
                pid = str(p.get("player_id", ""))
                if pid:
                    lineup_players[pid] = {
                        "name": p.get("name", ""),
                        "jersey_number": p.get("shirt_number") or None,
                        "position": p.get("position") or None,
                        "team": team_of_lineup,
                    }

        # Wyscout events carry team ids, not names — first-seen team id
        # that matches a lineup team wins; otherwise first-seen = home.
        team_id_to_side: dict[str, str] = {}
        for lu in lineups:
            tid = str(lu.get("team_id", ""))
            if tid and tid not in team_id_to_side:
                team_id_to_side[tid] = (
                    "home"
                    if lu.get("team_name") == home
                    else "away"
                    if lu.get("team_name") == away
                    else ""
                ) or ("home" if "home" not in team_id_to_side.values() else "away")

        imported = 0
        skipped = 0
        players_registered = 0
        shots = 0
        goals = 0
        total_xg = 0.0
        seen: set[str] = set()
        track_ids: dict[str, int] = {}

        for wev in events:
            tid_str = str(wev.team_id)
            if tid_str not in team_id_to_side:
                team_id_to_side[tid_str] = (
                    "home" if "home" not in team_id_to_side.values() else "away"
                )
            team = team_id_to_side[tid_str]

            player_key = str(wev.player_id)
            info = lineup_players.get(player_key, {})
            player_name = info.get("name") or (f"Wyscout {player_key}" if player_key else "")
            if player_key and player_key not in seen:
                seen.add(player_key)
                new_tid = self._stable_track_id(team, player_name or player_key)
                await self.storage.save_player(
                    match_id,
                    {
                        "track_id": new_tid,
                        "name": player_name or f"Player {player_key}",
                        "team": info.get("team") or team,
                        "position": info.get("position"),
                        "jersey_number": info.get("jersey_number"),
                    },
                )
                track_ids[player_key] = new_tid
                players_registered += 1

            ktype = _WYSCOUT_TYPE_MAP.get(wev.event_type, wev.event_type.lower().replace(" ", "_"))
            sx, sy = _norm100_to_meters(wev.x, wev.y)
            ex, ey = _norm100_to_meters(wev.end_x, wev.end_y)
            meta: dict[str, Any] = {
                "start_x": round(sx, 2),
                "start_y": round(sy, 2),
                "end_x": round(ex, 2),
                "end_y": round(ey, 2),
                "wyscout_event_id": wev.event_id,
                "wyscout_tags": wev.tags,
                "source": "wyscout",
            }
            is_shot = ktype == "shot" or "shot" in wev.event_type.lower()
            if is_shot:
                shots += 1
                tag_names = {str(t).lower() for t in wev.tags}
                meta["is_goal"] = "goal" in tag_names or "101" in tag_names
                if meta["is_goal"]:
                    goals += 1
                meta["shot_outcome"] = "goal" if meta["is_goal"] else "off_target"
                meta["xg"] = 0.0

            saved = await self.storage.save_event(
                match_id,
                {
                    "type": ktype,
                    "timestamp": float(wev.minute) * 60.0 + float(wev.second),
                    "team": team,
                    "completed": True,
                    "from_track_id": track_ids.get(player_key),
                    "metadata": meta,
                },
            )
            if saved:
                imported += 1
            else:
                skipped += 1

        await self._cache_totals(match_id, imported, shots, goals, total_xg)
        return {
            "match_id": match_id,
            "match_name": name,
            "vendor": "wyscout",
            "events_imported": imported,
            "events_skipped": skipped,
            "players_registered": players_registered,
            "shots": shots,
            "goals": goals,
            "xg_total": round(total_xg, 3),
            "source_file": str(path),
        }

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _stable_track_id(team_name: str, player_name: str) -> int:
        """Deterministic per-(team, player) track id in [1, 999] — same
        convention as StatsBombImportService."""
        return (abs(hash((team_name, player_name))) % 998) + 1

    async def _cache_totals(
        self,
        match_id: int,
        events: int,
        shots: int,
        goals: int,
        total_xg: float,
    ) -> None:
        await self.storage.save_advanced_metrics(
            match_id, "events_total", float(events), metric_category="import"
        )
        await self.storage.save_advanced_metrics(
            match_id, "shots_total", float(shots), metric_category="import"
        )
        await self.storage.save_advanced_metrics(
            match_id, "goals_total", float(goals), metric_category="import"
        )
        if total_xg:
            await self.storage.save_advanced_metrics(
                match_id, "xg_total", round(total_xg, 3), metric_category="import"
            )
