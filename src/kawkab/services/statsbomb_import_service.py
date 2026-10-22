"""StatsBomb event-data import — the elite-club interop path.

Lets a club point Kawkab at an existing StatsBomb events JSON file (open
data or licensed exports) and get a fully-populated Kawkab match —
events, players, teams, xG already computed by the trained model — with
NO video capture. Every downstream model in the 117-module core stack
then runs against it.

Design notes:
    - Coordinates are normalized to Kawkab meters (105x68) in a single
      attacking frame, same as core/validation/statsbomb_loader.py.
    - Both teams' events are imported; ``team`` is set to "home"/"away"
      by the possession_team vs match home/away comparison.
    - The shooter/passer is stored as a synthetic per-match player row
      (track_id = stable per-match jersey index) so player-level models
      (goals_added, pass networks, ...) work unchanged.
    - Match-level totals (xG, shots, goals) are cached into
      advanced_metrics the same way the video pipeline does.

Usage:
    from kawkab.services.statsbomb_import_service import StatsBombImportService
    svc = StatsBombImportService(storage_service)
    match_id = await svc.import_match("data/statsbomb_corpus/18245.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kawkab.core.validation.statsbomb_loader import (
    sb_to_meters,
    shot_deviation_angle,
    shot_distance_angle,
)
from kawkab.core.xg_model import active_xg_model
from kawkab.services.storage_errors import StorageDuplicateError

logger = logging.getLogger(__name__)

# StatsBomb event type -> Kawkab event_type (subset that maps cleanly;
# unmapped types are preserved with their original name lowercased so
# nothing is silently dropped).
_SB_TYPE_MAP: dict[str, str | None] = {
    "Pass": "pass",
    "Shot": "shot",
    "Carry": "carry",
    "Dribble": "dribble",
    "Duel": "duel",
    "Interception": "interception",
    "Clearance": "clearance",
    "Goal Keeper": "save",
    "Foul Committed": "foul",
    "Foul Won": "foul_won",
    "Miscontrol": "miscontrol",
    "Ball Receipt*": "ball_receipt",
    "Dispossessed": "dispossessed",
    "Aerial Lost": "aerial_lost",
    "Aerial Won": "aerial_won",
    "Offside": "offside",
    "Injury Stoppage": "stoppage",
    "Substitution": "substitution",
    "Tactical Shift": "tactical_shift",
    "Half Start": "half_start",
    "Half End": "half_end",
    "Starting XI": "starting_xi",
    "Camera Flash": None,  # broadcast noise, not a football action
    "Referee Ball-Drop": "referee_ball_drop",
    # Complete census of remaining SB open-data types (verified against the
    # 297-match corpus): mapped to the core/ conventions the recovery /
    # pressing / duel modules expect.
    "Pressure": "pressure",
    "Ball Recovery": "interception",  # SB 'Ball Recovery' = won loose ball;
    #   core/ RECOVERY_EVENT_TYPES uses {interception, tackle, loose_ball,
    #   goal_kick, clearance} — mapping to interception keeps it countable
    #   there without inventing a new event type the models don't read.
    "Block": "block",
    "Dribbled Past": "dribbled_past",
    "Shield": "shield",
    "50/50": "duel",  # contested loose ball — duel-type module reads it
    "Error": "miscontrol",  # SB 'Error' = unforced loss of possession
    "Bad Behaviour": "foul",
    "Player Off": "player_off",
    "Player On": "player_on",
    "Own Goal Against": "own_goal",
    "Own Goal For": "own_goal_won",
}

# Events imported for analytics (skip pure metadata rows entirely)
_SKIPPED_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Substitution",
    "Injury Stoppage",
    "Camera Flash",
}

ON_TARGET_OUTCOMES = {"Saved", "Goal", "Post", "Saved to Post"}
SHOT_OUTCOME_MAP = {
    "Goal": "goal",
    "Saved": "saved",
    "Blocked": "blocked",
    "Off T": "off_target",
    "Post": "post",
    "Wayward": "off_target",
    "Saved to Post": "saved",
    "Saved Off Target": "saved",
}


class StatsBombImportService:
    """Imports a StatsBomb events file into the Kawkab database."""

    def __init__(self, storage_service: Any) -> None:
        self.storage = storage_service
        self._xg_model = active_xg_model()

    async def import_match(
        self,
        events_path: str | Path,
        *,
        match_name: str | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
    ) -> dict[str, Any]:
        """Import one match. Returns a summary dict (match_id, counts)."""
        events_path = Path(events_path)
        with open(events_path) as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{events_path}: expected a non-empty JSON event list")

        # Home team = first "Starting XI"/period-1 possession team heuristic:
        # SB doesn't label home/away in the events file itself; the first
        # event's `team` is conventionally the home side in open data.
        first_team = (raw[0].get("team") or {}).get("name") or "Home"
        second_team = next(
            (
                (e.get("team") or {}).get("name")
                for e in raw
                if (e.get("team") or {}).get("name") not in (None, first_team)
            ),
            "Away",
        )
        home = home_team or first_team
        away = away_team or second_team
        name = match_name or f"{home} vs {away} (StatsBomb import)"

        # 1. Create the match (video_path: none — pure event-data import)
        match_id = await self.storage.save_match(
            name=name, video_path="", home_team=home, away_team=away
        )
        if not match_id:
            raise RuntimeError("storage refused to create the match row")

        # 2. Register players (stable synthetic track ids: hashed jersey)
        player_ids: dict[tuple[str, str], int] = {}
        seen: set[tuple[str, str]] = set()
        imported_players = 0

        # 3. Events
        imported_events = 0
        skipped = 0
        goals = 0
        shots = 0
        total_xg = 0.0

        for ev in raw:
            type_name = ev.get("type", {}).get("name", "")
            if type_name in _SKIPPED_TYPES:
                continue
            team_name = (ev.get("team") or {}).get("name") or ""
            team = "home" if team_name == home else "away"

            player = ev.get("player") or {}
            player_name = player.get("name", "")
            if player_name:
                key = (team_name, player_name)
                if key not in seen:
                    seen.add(key)
                    track_id = self._stable_track_id(team_name, player_name)
                    await self.storage.save_player(
                        match_id,
                        {
                            "track_id": track_id,
                            "name": player_name,
                            "team": team,
                            "position": self._sb_position_name(ev.get("position")),
                            "jersey_number": None,
                        },
                    )
                    player_ids[key] = track_id
                    imported_players += 1

            kawkab_event = self._convert_event(ev, team, player_ids)
            if kawkab_event is None:
                skipped += 1
                continue
            if kawkab_event["type"] == "shot":
                shots += 1
                if kawkab_event["metadata"].get("is_goal"):
                    goals += 1
                total_xg += kawkab_event["metadata"].get("xg", 0.0)
            from_track = player_ids.get((team_name, player_name))
            if from_track is not None:
                kawkab_event["from_track_id"] = from_track
            try:
                await self.storage.save_event(match_id, kawkab_event)
            except StorageDuplicateError:
                # The events table has a dedup unique index (migration 015)
                # on (match_id, timestamp, event_type, from_track_id) —
                # two same-type events by the same player in the same
                # second (e.g. consecutive ball receipts) are true
                # duplicates under that key. Skip them, count as skipped.
                # Typed duplicate: works on BOTH adapters (the raw
                # sqlite3.IntegrityError this replaces never matched
                # Postgres's asyncpg error, so PG re-imports crashed).
                skipped += 1
                continue
            imported_events += 1

        # 4. Cache headline metrics the way the video pipeline does
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
            "events_imported": imported_events,
            "events_skipped": skipped,
            "players_registered": imported_players,
            "shots": shots,
            "goals": goals,
            "xg_total": round(total_xg, 3),
            "source_file": str(events_path),
        }

    # ── helpers ────────────────────────────────────────────────────────────

    def _stable_track_id(self, team_name: str, player_name: str) -> int:
        """Deterministic per-(team, player) track id in [1, 999]."""
        return (abs(hash((team_name, player_name))) % 998) + 1

    def _sb_position_name(self, position: dict | None) -> str:
        name = (position or {}).get("name", "")
        mapping = {
            "Goalkeeper": "GK",
            "Right Center Back": "CB",
            "Left Center Back": "CB",
            "Center Back": "CB",
            "Right Back": "RB",
            "Left Back": "LB",
            "Right Wing Back": "RWB",
            "Left Wing Back": "LWB",
            "Defensive Midfield": "DM",
            "Right Center Midfield": "CM",
            "Left Center Midfield": "CM",
            "Center Midfield": "CM",
            "Right Midfield": "RM",
            "Left Midfield": "LM",
            "Right Wing": "RW",
            "Left Wing": "LW",
            "Center Attacking Midfield": "AM",
            "Right Attacking Midfield": "AM",
            "Left Attacking Midfield": "AM",
            "Center Forward": "CF",
            "Left Center Forward": "CF",
            "Right Center Forward": "CF",
            "Striker": "ST",
            "Secondary Striker": "SS",
        }
        return mapping.get(name, "MID")

    def _convert_event(
        self,
        ev: dict[str, Any],
        team: str,
        player_ids: dict[tuple[str, str], int],
    ) -> dict[str, Any] | None:
        """One StatsBomb event -> one Kawkab event dict (or None to skip)."""
        type_name = ev.get("type", {}).get("name", "")
        kawkab_type = _SB_TYPE_MAP.get(type_name, type_name.lower() or None)
        if kawkab_type is None:
            return None

        loc = ev.get("location") or []
        meta: dict[str, Any] = {}
        timestamp = self._event_timestamp(ev)

        start_x_m, start_y_m = (None, None)
        if len(loc) >= 2:
            start_x_m, start_y_m = sb_to_meters(loc)
            meta["start_x"] = round(start_x_m, 2)
            meta["start_y"] = round(start_y_m, 2)

        completed = True
        end_x_m = end_y_m = None

        if type_name == "Pass":
            pass_data = ev.get("pass") or {}
            end_loc = pass_data.get("end_location") or []
            if len(end_loc) >= 2:
                end_x_m, end_y_m = sb_to_meters(end_loc)
                meta["end_x"] = round(end_x_m, 2)
                meta["end_y"] = round(end_y_m, 2)
            completed = pass_data.get("outcome") is None
            meta["pass_length_m"] = round(pass_data.get("length", 0.0) * 0.875, 2)
            height = (pass_data.get("height") or {}).get("name", "")
            if height:
                meta["pass_height"] = height.lower()
            body = (pass_data.get("body_part") or {}).get("name", "")
            if body:
                meta["body_part"] = body.lower().replace(" foot", "_foot").replace(" ", "_")

        elif type_name == "Shot":
            shot = ev.get("shot") or {}
            outcome_name = shot.get("outcome", {}).get("name", "")
            meta["shot_outcome"] = SHOT_OUTCOME_MAP.get(outcome_name, outcome_name.lower())
            meta["is_goal"] = outcome_name == "Goal"
            meta["statsbomb_xg"] = float(shot.get("statsbomb_xg", 0.0) or 0.0)
            if start_x_m is not None and start_y_m is not None:
                distance_m, angle_deg = shot_distance_angle(start_x_m, start_y_m)
                # Two angle conventions on purpose (2026-09-16):
                #   angle_deg         = goal-OPENING angle  -> PSxG serving model
                #   angle_deviation_deg = deviation-from-central -> xG serving model
                # The xG model's (1 - cos(angle)) feature is calibrated for the
                # deviation convention; feeding it the opening angle trained/
                # scored every angle term backwards (see
                # statsbomb_loader.shot_deviation_angle).
                angle_deviation_deg = shot_deviation_angle(start_x_m, start_y_m)
                meta["distance_m"] = round(distance_m, 2)
                meta["angle_deg"] = round(angle_deg, 2)
                meta["angle_deviation_deg"] = round(angle_deviation_deg, 2)
                body = (shot.get("body_part") or {}).get("name", "Right Foot")
                shot_type = (shot.get("type") or {}).get("name", "Open Play")
                meta["xg"] = round(
                    self._xg_model.compute(
                        {
                            "type": "shot",
                            "distance_m": distance_m,
                            "angle_deg": angle_deviation_deg,
                            "body_part": {
                                "Right Foot": "right_foot",
                                "Left Foot": "left_foot",
                                "Head": "head",
                            }.get(body, "right_foot"),
                            "shot_type": {
                                "Open Play": "open_play",
                                "Free Kick": "free_kick",
                                "Penalty": "penalty",
                            }.get(shot_type, "open_play"),
                        }
                    ),
                    4,
                )
            end_loc = shot.get("end_location") or []
            if len(end_loc) >= 2:
                ex, ey = sb_to_meters(end_loc[:2])
                meta["end_x"] = round(ex, 2)
                meta["end_y"] = round(ey, 2)
            if len(end_loc) >= 3:
                meta["end_z"] = round(end_loc[2], 2)

        elif type_name == "Carry":
            carry = ev.get("carry") or {}
            end_loc = carry.get("end_location") or []
            if len(end_loc) >= 2:
                end_x_m, end_y_m = sb_to_meters(end_loc)
                meta["end_x"] = round(end_x_m, 2)
                meta["end_y"] = round(end_y_m, 2)

        elif type_name == "Dribble":
            dribble = ev.get("dribble") or {}
            completed = dribble.get("outcome", {}).get("name") == "Complete"

        elif type_name == "Duel":
            duel = ev.get("duel") or {}
            completed = duel.get("outcome", {}).get("name") == "Won"

        elif type_name == "Goal Keeper":
            gk = ev.get("goalkeeper") or {}
            gk_type = (gk.get("type") or {}).get("name", "")
            meta["gk_action"] = gk_type.lower().replace(" ", "_")

        event = {
            "type": kawkab_type,
            "timestamp": timestamp,
            "team": team,
            "completed": completed,
            "confidence": 1.0,  # human-tagged professional data
            "metadata": meta,
        }
        if "minute" in ev and "second" in ev:
            event["minute"] = ev.get("minute")
            event["second"] = ev.get("second")
        return event

    def _event_timestamp(self, ev: dict[str, Any]) -> float:
        """Seconds from kickoff; SB 'timestamp' is ISO wall-clock."""
        period = int(ev.get("period", 1) or 1)
        minute = int(ev.get("minute", 0) or 0)
        second = int(ev.get("second", 0) or 0)
        base = 2700.0 if period == 2 else 0.0
        return base + minute * 60.0 + second
