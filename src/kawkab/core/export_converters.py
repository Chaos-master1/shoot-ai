"""Export converters — transforms internal events to professional formats.

Supports:
- StatsBomb JSON format (used by StatsBomb, Wyscout)
- SPADL event stream format (used by socceraction, VAEP)
- Opta-style CSV format (used by Opta, WhoScored)
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any

from kawkab.core.events import (
    BodyPart,
    CarryEvent,
    PassEvent,
    PassType,
    ShotEvent,
    ShotType,
    event_from_dict,
)


def _norm_to_meters(val: float, pitch_dim: float) -> float:
    """Convert normalized coord to meters if coords are normalized (0-1)."""
    return val if val > 1.5 else val * pitch_dim


def to_statsbomb_json(
    events: list[dict[str, Any]],
    match_id: int = 0,
    home_team: str = "Home",
    away_team: str = "Away",
    home_team_id: int = 1,
    away_team_id: int = 2,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> str:
    """Convert match events to StatsBomb JSON format.

    Produces an array of events matching the StatsBomb open data
    specification. Each event includes 'type', 'timestamp', 'period',
    'location' [x, y], 'player', 'team', and type-specific fields.

    Coordinate auto-detection: if values > 1.5 they're already in meters,
    otherwise they're normalized (0-1) and multiplied by pitch dimensions.
    """
    sb_events: list[dict[str, Any]] = []

    team_map = {
        "home": {"id": home_team_id, "name": home_team},
        "away": {"id": away_team_id, "name": away_team},
    }

    for raw in events:
        try:
            ev = event_from_dict(raw)
        except Exception:
            ev_type = raw.get("type", "pass")
            ev = ShotEvent.from_dict(raw) if ev_type == "shot" else PassEvent.from_dict(raw)

        team_info = team_map.get(ev.team, {"id": 0, "name": ev.team})
        period = 1 if ev.timestamp < (45 * 60) else 2
        ts_seconds = ev.timestamp
        minutes = int(ts_seconds // 60)
        seconds = int(ts_seconds % 60)
        timestamp_str = f"{minutes:02d}:{seconds:02d}"

        location: list[float] | None = None
        if ev.x is not None and ev.y is not None:
            location = [ev.x, ev.y]

        sb_event: dict[str, Any] = {
            "id": len(sb_events) + 1,
            "match_id": match_id,
            "team": {"id": team_info["id"], "name": team_info["name"]},
            "timestamp": timestamp_str,
            "minute": minutes,
            "second": seconds,
            "period": period,
            "event_type": ev.type.value,
            "location": location,
        }

        if ev.track_id is not None:
            sb_event["player"] = {"id": ev.track_id, "name": f"Player #{ev.track_id}"}

        if isinstance(ev, PassEvent):
            sb_event["pass"] = {
                "length": round(ev.length_m, 1),
                "angle": 0.0,
                "height": "ground" if ev.body_part != BodyPart.HEAD else "high",
                "body_part": {"id": _body_part_id(ev.body_part), "name": ev.body_part.value},
                "recipient": {"id": ev.to_track_id, "name": f"Player #{ev.to_track_id}"}
                if ev.to_track_id
                else None,
                "pass_type": {"id": _pass_type_id(ev.pass_type), "name": ev.pass_type.value},
                "through_ball": ev.is_through_ball,
                "cross": ev.is_cross,
                "switch": ev.is_switch,
                "outcome": {"id": 1, "name": "Complete"}
                if ev.completed
                else {"id": 2, "name": "Incomplete"},
            }
            if ev.start_x is not None and ev.start_y is not None:
                sb_event["location"] = [
                    _norm_to_meters(ev.start_x, pitch_length_m),
                    _norm_to_meters(ev.start_y, pitch_width_m),
                ]
                sb_event["pass"]["end_location"] = [
                    _norm_to_meters(ev.end_x or 0.5, pitch_length_m),
                    _norm_to_meters(ev.end_y or 0.5, pitch_width_m),
                ]
        elif isinstance(ev, ShotEvent):
            sb_event["shot"] = {
                "xg": round(ev.xg, 4),
                "statsbomb_xg": round(ev.xg, 4),
                "body_part": {"id": _body_part_id(ev.body_part), "name": ev.body_part.value},
                "type": {"id": _shot_type_id(ev.shot_type), "name": ev.shot_type.value},
                "outcome": {"id": 1, "name": "On Target"}
                if ev.on_target
                else {"id": 2, "name": "Off Target"},
                "one_on_one": ev.is_one_on_one,
            }
        elif isinstance(ev, CarryEvent):
            sb_event["carry"] = {
                "length": round(ev.distance_m, 1),
                "progressive": ev.is_progressive,
            }
            if ev.start_x and ev.start_y:
                sb_event["location"] = [ev.start_x, ev.start_y]
                sb_event["carry"]["end_location"] = [ev.end_x or 0, ev.end_y or 0]

        sb_events.append(sb_event)

    return json.dumps(sb_events, indent=2)


def to_spadl_csv(
    events: list[dict[str, Any]],
    match_id: int = 0,
    home_team_id: int = 1,
    away_team_id: int = 2,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> str:
    """Convert match events to SPADL CSV format.

    SPADL (Soccer Player Action Description Language) encodes each
    action as a row with: game_id, period_id, time_seconds, team_id,
    player_id, start_x, start_y, end_x, end_y, action_id, bodypart_id,
    result_id. Compatible with the socceraction library.
    """
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "game_id",
            "period_id",
            "time_seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "action_id",
            "action_name",
            "bodypart_id",
            "result_id",
        ]
    )

    spadl_actions = {
        "pass": 1,
        "shot": 2,
        "carry": 3,
        "tackle": 4,
        "interception": 5,
        "foul": 6,
    }

    team_ids = {"home": home_team_id, "away": away_team_id}

    for raw in events:
        try:
            ev = event_from_dict(raw)
        except Exception:
            continue

        period = 1 if ev.timestamp < (45 * 60) else 2
        team_id = team_ids.get(ev.team, 0)
        action_id = spadl_actions.get(ev.type.value, 0)

        start_x = 0.0
        start_y = 0.0
        end_x = 0.0
        end_y = 0.0
        bodypart_id = 1

        if isinstance(ev, PassEvent):
            start_x = _norm_to_meters(ev.start_x or 0.5, pitch_length_m)
            start_y = _norm_to_meters(ev.start_y or 0.5, pitch_width_m)
            end_x = _norm_to_meters(ev.end_x or 0.5, pitch_length_m)
            end_y = _norm_to_meters(ev.end_y or 0.5, pitch_width_m)
            bodypart_id = _body_part_id(ev.body_part)
        elif isinstance(ev, ShotEvent):
            goal_x = 0 if ev.x is not None and ev.x < pitch_length_m / 2 else pitch_length_m
            goal_y = pitch_width_m / 2
            start_x = ev.x or 0
            start_y = ev.y or 0
            end_x = goal_x
            end_y = goal_y
            bodypart_id = _body_part_id(ev.body_part)
        elif isinstance(ev, CarryEvent):
            start_x = ev.start_x or 0
            start_y = ev.start_y or 0
            end_x = ev.end_x or 0
            end_y = ev.end_y or 0
            bodypart_id = _body_part_id(ev.body_part)

        result_id = 1  # success
        if isinstance(ev, PassEvent) and not ev.completed:
            result_id = 0

        writer.writerow(
            [
                match_id,
                period,
                round(ev.timestamp, 1),
                team_id,
                ev.track_id or 0,
                round(start_x, 1),
                round(start_y, 1),
                round(end_x, 1),
                round(end_y, 1),
                action_id,
                ev.type.value,
                bodypart_id,
                result_id,
            ]
        )

    return output.getvalue()


def to_opta_csv(
    events: list[dict[str, Any]],
    match_id: int = 0,
) -> str:
    """Export events in Opta-style CSV format. Compatible with
    common football analytics tools."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "match_id",
            "event_id",
            "type",
            "period",
            "minute",
            "second",
            "team",
            "player_id",
            "x",
            "y",
            "end_x",
            "end_y",
            "outcome",
            "value",
        ]
    )

    for i, raw in enumerate(events):
        try:
            ev = event_from_dict(raw)
        except Exception:
            continue

        period = 1 if ev.timestamp < (45 * 60) else 2
        minute = int(ev.timestamp // 60)
        second = int(ev.timestamp % 60)

        x = ev.x or 0
        y = ev.y or 0
        end_x = 0.0
        end_y = 0.0
        outcome = 1
        value = 0.0

        if isinstance(ev, PassEvent):
            end_x = (ev.end_x or 0.5) * 100
            end_y = (ev.end_y or 0.5) * 100
            if not ev.completed:
                outcome = 0
            if ev.is_key_pass:
                value = 1
            if ev.is_assist:
                value = 2
        elif isinstance(ev, ShotEvent):
            end_x = 0
            end_y = 0
            outcome = 1 if ev.on_target else 0
            value = ev.xg
        elif isinstance(ev, CarryEvent):
            end_x = float(ev.end_x) if ev.end_x else 0.0
            end_y = float(ev.end_y) if ev.end_y else 0.0
            value = ev.distance_m

        # Convert to 0-100 pitch coordinates (Opta style)
        writer.writerow(
            [
                match_id,
                i + 1,
                ev.type.value,
                period,
                minute,
                second,
                ev.team,
                ev.track_id or 0,
                round(x, 1),
                round(y, 1),
                round(end_x, 1),
                round(end_y, 1),
                outcome,
                round(value, 4),
            ]
        )

    return output.getvalue()


def _body_part_id(bp: BodyPart) -> int:
    mapping = {
        BodyPart.RIGHT_FOOT: 1,
        BodyPart.LEFT_FOOT: 2,
        BodyPart.HEAD: 3,
        BodyPart.OTHER: 4,
    }
    return mapping.get(bp, 1)


def _pass_type_id(pt: PassType) -> int:
    mapping = {
        PassType.STANDARD: 1,
        PassType.THROUGH_BALL: 2,
        PassType.CROSS: 3,
        PassType.SWITCH: 4,
        PassType.HEADER: 5,
        PassType.LONG_BALL: 6,
        PassType.BACK_PASS: 7,
        PassType.ONE_TOUCH: 8,
        PassType.FLICK_ON: 9,
    }
    return mapping.get(pt, 1)


def _shot_type_id(st: ShotType) -> int:
    mapping = {
        ShotType.OPEN_PLAY: 1,
        ShotType.FREE_KICK: 2,
        ShotType.PENALTY: 3,
        ShotType.HEADER: 4,
        ShotType.VOLLEY: 5,
        ShotType.HALF_VOLLEY: 6,
    }
    return mapping.get(st, 1)
