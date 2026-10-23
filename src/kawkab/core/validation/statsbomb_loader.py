"""StatsBomb open-data event loader — converts to Kawkab event schema.

StatsBomb pitch conventions (v1 open data):
    - x in [0, 120], y in [0, 80]  (1 unit ≈ 0.9144 m along x — the pitch
      is 120x80 units mapped onto a ~105x68 m pitch)
    - origin (0,0) at the *attacking-team's* left corner as broadcast;
      x=0 is each team's own goal line, x increases toward the opponent's
      goal. Both teams shoot toward x=120 *in their own possession frame*.

Kawkab convention (see ``core/events.py`` / ``game_constants.py``):
    - meters, x in [0, 105], y in [0, 68], origin at the home team's goal
      line left corner, y measured from one touchline.

This loader always normalizes StatsBomb coordinates to Kawkab meters and
returns shots in a single attacking-direction frame (toward x=105) so
distance/angle/GK features are direction-correct for every shot.

No I/O at import time. Only stdlib + numpy.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kawkab.core.player_positions import position_code

logger = logging.getLogger(__name__)

# ── Coordinate constants ─────────────────────────────────────────────────────

# StatsBomb grid -> meters (open data spec: 120x80 units ≈ 105x68 m)
SB_X_MAX = 120.0
SB_Y_MAX = 80.0
KAWKAB_X_MAX = 105.0
KAWKAB_Y_MAX = 68.0
UNIT_TO_M_X = KAWKAB_X_MAX / SB_X_MAX  # 0.875
UNIT_TO_M_Y = KAWKAB_Y_MAX / SB_Y_MAX  # 0.85

# Kawkab goal geometry (meters) — matches core/game_constants.py
GOAL_WIDTH_M = 7.32
HALF_GOAL_W = GOAL_WIDTH_M / 2.0
GOAL_CENTER_Y = KAWKAB_Y_MAX / 2.0  # 34.0
GOAL_LINE_X = KAWKAB_X_MAX  # shots attack toward x=105

# ── Event-type mapping ──────────────────────────────────────────────────────

SHOT_TYPE_MAP = {
    "Open Play": "open_play",
    "Free Kick": "free_kick",
    "Penalty": "penalty",
    "Corner": "corner",  # shots direct from corners
    "Kick Off": "open_play",
}

TECHNIQUE_VOLLEY = {"Volley", "Half Volley"}
OUTCOME_GOAL = "Goal"


def sb_to_meters(loc: list[float] | tuple[float, float]) -> tuple[float, float]:
    """Convert StatsBomb (x, y) units to Kawkab meters."""
    x_m = loc[0] * UNIT_TO_M_X
    y_m = loc[1] * UNIT_TO_M_Y
    return x_m, y_m


def shot_deviation_angle(x_m: float, y_m: float) -> float:
    """Deviation-from-central angle (deg) — EnhancedXgModel's convention.

    0 deg = dead central (the goal-center line), approaching 90 deg =
    along the goal line to the side. cos(deviation) is the visible-goal
    fraction the serving model consumes via its (1 - cos) angle feature.

    History (2026-09-16): ``shot_distance_angle`` below returns the
    goal-OPENING angle (posts subtended from the shot — central ~36 deg,
    goal-line-wide ~90+). Feeding that into the xG trainer trained every
    angle coefficient BACKWARDS (wide shots outscoring central ones;
    distance decay collapsed onto the angle term) — the exact bug the
    56-shot fit in scripts/train_xg_from_statsbomb.py had, reintroduced
    through this loader. That script's own _get_angle already carries the
    corrected deviation convention and its docstring documents the same
    history. Opening angle stays available (and is correct) for PSxG.
    """
    dx = GOAL_LINE_X - x_m
    dy = GOAL_CENTER_Y - y_m
    dist = math.hypot(dx, dy)
    if dist < 0.5:
        return 90.0
    return min(math.degrees(math.atan2(abs(dy), abs(dx))), 90.0)


def shot_distance_angle(x_m: float, y_m: float) -> tuple[float, float]:
    """Distance (m) and opening angle (deg) from a shot position to the goal.

    The goal is at (105, 34); angle is the opening angle subtended by the
    goal mouth at the shot position, in degrees [0, 90]. PSxG's
    ``angle_opening_deg`` feature expects exactly this convention — do
    not "fix" it toward deviation-from-central (see
    ``shot_deviation_angle`` for the xG-convention counterpart).
    """
    dx = GOAL_LINE_X - x_m
    dy = GOAL_CENTER_Y - y_m
    distance = math.hypot(dx, dy)
    if distance < 0.1:
        return 0.0, 90.0
    # Opening angle: angle between the two goalposts as seen from the shot.
    # cos(angle) via law of cosines on the (post, shot, post) triangle.
    a = math.hypot(GOAL_LINE_X - x_m, GOAL_CENTER_Y - HALF_GOAL_W - y_m)
    b = math.hypot(GOAL_LINE_X - x_m, GOAL_CENTER_Y + HALF_GOAL_W - y_m)
    cos_c = (a * a + b * b - GOAL_WIDTH_M * GOAL_WIDTH_M) / (2.0 * a * b)
    cos_c = max(-1.0, min(1.0, cos_c))
    angle_deg = math.degrees(math.acos(cos_c))
    return distance, angle_deg


def _freeze_frame_gk_distance(
    freeze_frame: list[dict[str, Any]] | None,
    shot_x_m: float,
    shot_y_m: float,
    shot_team_sb_id: int | None,
) -> float:
    """Distance from the *opposing* goalkeeper to the shot location (m).

    StatsBomb freeze frames list both teams with a ``teammate`` flag
    relative to the shot-taker's team; we want the goalkeeper whose
    ``teammate`` is False. Returns 0.0 when no freeze frame exists —
    the model treats 0 as "feature absent" (its gk_distance terms are
    only added when > 0), preserving the trained model's semantics.
    """
    if not freeze_frame:
        return 0.0
    best: float | None = None
    for p in freeze_frame:
        pos = p.get("position") or {}
        if pos.get("name") != "Goalkeeper":
            continue
        if p.get("teammate", False):
            continue
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue
        gx, gy = sb_to_meters(loc)
        d = math.hypot(gx - shot_x_m, gy - shot_y_m)
        if best is None or d < best:
            best = d
    return best if best is not None else 0.0


def _freeze_frame_pressure(freeze_frame: list[dict[str, Any]] | None) -> bool:
    """True when an opposing outfield player is within 1.5 m of a teammate.

    Freeze frames don't tag the shooter explicitly. Approximation (documented
    in the model card): mark pressed when any opposing player stands within
    1.5 m of any teammate — conservative, honest about its limits.
    """
    if not freeze_frame:
        return False
    opponents = [p for p in freeze_frame if not p.get("teammate", False)]
    teammates = [p for p in freeze_frame if p.get("teammate", False)]
    for op in opponents:
        ol = op.get("location")
        if not ol:
            continue
        ox, oy = sb_to_meters(ol)
        for tm in teammates:
            tl = tm.get("location")
            if not tl:
                continue
            tx, ty = sb_to_meters(tl)
            if math.hypot(ox - tx, oy - ty) <= 1.5:
                return True
    return False


@dataclass
class StatsBombShot:
    """One shot in Kawkab's feature schema + StatsBomb reference fields."""

    # Kawkab-native features (what the model consumes)
    distance_m: float
    angle_deg: float  # goal-OPENING angle (PSxG convention)
    body_part: str  # "right_foot" | "left_foot" | "head" | "other"
    shot_type: str  # "open_play" | "free_kick" | "penalty" | "corner"
    gk_distance_m: float
    is_pressed: bool
    is_one_on_one: bool
    is_rebound: bool
    is_big_chance: bool
    is_goal: bool

    # Ground-truth / reference fields (never model inputs)
    statsbomb_xg: float
    match_id: str
    event_id: str
    minute: int
    period: int
    player_name: str
    team_name: str

    # Deviation-from-central angle (xG convention, 0 = dead central).
    # Appended last so positional constructions of the original fields
    # keep their meaning; always computed by _parse_shot.
    angle_deviation_deg: float

    # StatsBomb's player id (0 when the event carries no player). Appended
    # with a default so existing keyword constructions keep working.
    player_id: int = 0

    def to_fit_dict(self) -> dict[str, Any]:
        """Dict suitable for ``xg_trainer.FitShot`` construction / fitting."""
        return {
            "distance_m": self.distance_m,
            # xG features consume the DEVIATION convention (0 = central);
            # the opening angle would train every angle term backwards
            # (see shot_deviation_angle).
            "angle_deg": self.angle_deviation_deg,
            "is_header": self.body_part == "head",
            "is_one_on_one": self.is_one_on_one,
            "is_pressed": self.is_pressed,
            "is_volley": False,  # filled by caller from technique if desired
            "is_free_kick": self.shot_type == "free_kick",
            "gk_distance_m": self.gk_distance_m,
            "is_rebound": self.is_rebound,
            "is_big_chance": self.is_big_chance,
            "is_goal": self.is_goal,
        }


@dataclass
class StatsBombPlayerMatch:
    """One player's involvement in one match.

    Identity comes from StatsBomb's own player id; position and minutes
    come from the Starting XI / substitution / red-card events. ``minutes``
    is regulation or extra time (90/120), not wall-clock stoppage time,
    and is 0 when the lineup cannot establish when the player was on.
    """

    player_id: int
    name: str
    team: str
    side: str  # "home" | "away"
    position: str  # Kawkab code, "" when the source position is unknown
    jersey_number: int | None
    minutes_played: float
    started: bool


@dataclass
class StatsBombMatch:
    """Parsed StatsBomb match: Kawkab-schema shots + raw reference data."""

    match_id: str
    shots: list[StatsBombShot] = field(default_factory=list)

    # Normalized per-team event stream (style metrics). Kept alongside the
    # shots so one read of a corpus file serves both measurements.
    team_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Player-match identity, positions and minutes (player benchmark).
    players: list[StatsBombPlayerMatch] = field(default_factory=list)

    # Aggregate reference values for cross-model comparison
    total_sb_xg: float = 0.0
    total_goals: int = 0
    n_events: int = 0
    home_team: str = ""
    away_team: str = ""

    def __post_init__(self) -> None:
        if self.shots:
            self.total_sb_xg = sum(s.statsbomb_xg for s in self.shots)
            self.total_goals = sum(1 for s in self.shots if s.is_goal)


def load_statsbomb_match(path: str | Path) -> StatsBombMatch:
    """Parse one StatsBomb events JSON file into a StatsBombMatch."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of events, got {type(raw).__name__}")

    match_id = path.stem
    home_team, away_team = infer_home_away(raw)
    match = StatsBombMatch(
        match_id=match_id,
        n_events=len(raw),
        home_team=home_team,
        away_team=away_team,
    )

    for ev in raw:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        shot = _parse_shot(ev, match_id)
        if shot is not None:
            match.shots.append(shot)

    match.team_events = normalize_statsbomb_events(raw)
    match.players = extract_statsbomb_players(raw, home=home_team, away=away_team)
    match.total_sb_xg = sum(s.statsbomb_xg for s in match.shots)
    match.total_goals = sum(1 for s in match.shots if s.is_goal)
    return match


def infer_home_away(raw: list[dict[str, Any]]) -> tuple[str, str]:
    """StatsBomb files do not label home/away; the first team is the home side.

    This is the same convention the importer and the corpus index use, kept
    in one place so lineup side assignment cannot drift from match creation.
    """
    teams: list[str] = []
    for ev in raw:
        name = (ev.get("team") or {}).get("name") or ""
        if name and name not in teams:
            teams.append(name)
        if len(teams) == 2:
            break
    home = teams[0] if teams else "Home"
    away = teams[1] if len(teams) > 1 else "Away"
    return home, away


def extract_statsbomb_players(
    raw: list[dict[str, Any]],
    *,
    home: str | None = None,
    away: str | None = None,
) -> list[StatsBombPlayerMatch]:
    """Player-match identity, position and minutes from the raw event list.

    Identity is StatsBomb's player id, not a name that can collide. Minutes
    come from the Starting XI, substitutions and red cards; regulation is
    90 minutes and matches with a third/fourth period use 120. A player
    whose on-time cannot be established gets 0 minutes rather than a guess,
    and the benchmark then refuses to percentile them.

    This is the single owner of the minutes convention for both the
    importer and the offline cohort builder. Pure function.
    """
    inferred_home, inferred_away = infer_home_away(raw)
    home = home or inferred_home
    away = away or inferred_away
    side_by_team = {home: "home", away: "away"}

    has_extra_time = any(int(ev.get("period", 1) or 1) >= 3 for ev in raw)
    end_minute = 120.0 if has_extra_time else 90.0

    records: dict[int, dict[str, Any]] = {}

    def ensure(player: Any, team_name: str) -> dict[str, Any] | None:
        if not isinstance(player, dict):
            return None
        try:
            player_id = int(player.get("id") or 0)
        except (TypeError, ValueError):
            return None
        if player_id <= 0:
            return None
        record = records.setdefault(
            player_id,
            {
                "player_id": player_id,
                "name": str(player.get("name") or ""),
                "team": team_name,
                "side": side_by_team.get(team_name, ""),
                "position": "",
                "jersey_number": None,
                "on": None,
                "off": end_minute,
                "started": False,
            },
        )
        if not record["name"] and player.get("name"):
            record["name"] = str(player.get("name"))
        if not record["team"] and team_name:
            record["team"] = team_name
            record["side"] = side_by_team.get(team_name, "")
        return record

    for ev in raw:
        if not isinstance(ev, dict):
            continue
        type_name = (ev.get("type") or {}).get("name", "")
        team_name = (ev.get("team") or {}).get("name") or ""
        try:
            minute = float(ev.get("minute") or 0.0)
        except (TypeError, ValueError):
            minute = 0.0

        if type_name == "Starting XI":
            for entry in (ev.get("tactics") or {}).get("lineup") or []:
                record = ensure(entry.get("player"), team_name)
                if record is None:
                    continue
                record["started"] = True
                record["on"] = 0.0
                record["off"] = end_minute
                record["position"] = (
                    position_code((entry.get("position") or {}).get("name")) or record["position"]
                )
                jersey = entry.get("jersey_number")
                record["jersey_number"] = int(jersey) if jersey is not None else None

        elif type_name == "Substitution":
            substitution = ev.get("substitution") or {}
            off = ensure(ev.get("player"), team_name)
            if off is not None:
                off["off"] = min(float(off["off"]), minute)
            on = ensure(substitution.get("replacement"), team_name)
            if on is not None:
                on["on"] = minute if on["on"] is None else min(float(on["on"]), minute)
                on["off"] = end_minute

        elif type_name in ("Foul Committed", "Bad Behaviour"):
            card = ((ev.get(type_name.lower().replace(" ", "_")) or {}).get("card") or {}).get(
                "name", ""
            )
            if card in ("Red Card", "Second Yellow"):
                sent_off = ensure(ev.get("player"), team_name)
                if sent_off is not None:
                    sent_off["off"] = min(float(sent_off["off"]), minute)

        player = ev.get("player")
        if player:
            record = ensure(player, team_name)
            if record is not None and not record["position"]:
                record["position"] = position_code((ev.get("position") or {}).get("name"))

    players: list[StatsBombPlayerMatch] = []
    for record in records.values():
        if not record["name"]:
            continue
        on = record["on"]
        minutes = 0.0 if on is None else max(0.0, min(end_minute, float(record["off"]) - on))
        players.append(
            StatsBombPlayerMatch(
                player_id=record["player_id"],
                name=record["name"],
                team=record["team"],
                side=record["side"],
                position=record["position"],
                jersey_number=record["jersey_number"],
                minutes_played=round(minutes, 1),
                started=bool(record["started"]),
            )
        )

    players.sort(
        key=lambda p: (
            0 if p.side == "home" else 1,
            0 if p.started else 1,
            p.jersey_number if p.jersey_number is not None else 999,
            p.name,
            p.player_id,
        )
    )
    return players


def _parse_shot(ev: dict[str, Any], match_id: str) -> StatsBombShot | None:
    """Convert one StatsBomb 'Shot' event into a StatsBombShot."""
    shot = ev.get("shot") or {}
    loc = ev.get("location")
    if not loc or len(loc) < 2:
        return None

    x_m, y_m = sb_to_meters(loc)
    distance_m, angle_deg = shot_distance_angle(x_m, y_m)
    angle_deviation_deg = shot_deviation_angle(x_m, y_m)

    body_part_name = (shot.get("body_part") or {}).get("name", "Right Foot")
    body_part = {
        "Right Foot": "right_foot",
        "Left Foot": "left_foot",
        "Head": "head",
        "Other": "other",
        "No Contact": "other",
    }.get(body_part_name, "right_foot")

    shot_type_name = (shot.get("type") or {}).get("name", "Open Play")
    shot_type = SHOT_TYPE_MAP.get(shot_type_name, "open_play")

    technique = (shot.get("technique") or {}).get("name", "")

    outcome_name = (shot.get("outcome") or {}).get("name", "")
    is_goal = outcome_name == OUTCOME_GOAL

    # One-on-one: shooter is between the last outfield defender line and
    # the GK, and the GK is the nearest opponent. Approximation: nearest
    # opponent > 3 m away and GK within 12 m of the shot.
    gk_dist = _freeze_frame_gk_distance(
        shot.get("freeze_frame"), x_m, y_m, (ev.get("team") or {}).get("id")
    )
    # StatsBomb tags one-on-ones explicitly on the shot dict.
    one_on_one_raw = shot.get("one_on_one")
    if isinstance(one_on_one_raw, bool):
        is_one_on_one = one_on_one_raw
    else:
        is_one_on_one = bool(
            gk_dist > 0
            and gk_dist < 12.0
            and _nearest_opponent_distance(shot.get("freeze_frame"), loc) > 3.0
        )

    # Rebound: SB open data has no explicit flag; Dribble->Shot sequences
    # aren't tagged here. Keep the technique name as the only source.
    is_rebound = technique == "Rebound"

    # Big chance: not in open-data shot dicts; never fabricated.
    return StatsBombShot(
        distance_m=distance_m,
        angle_deg=angle_deg,
        angle_deviation_deg=angle_deviation_deg,
        body_part=body_part,
        shot_type=shot_type,
        gk_distance_m=gk_dist,
        is_pressed=_freeze_frame_pressure(shot.get("freeze_frame")),
        is_one_on_one=is_one_on_one,
        is_rebound=is_rebound,
        is_big_chance=False,
        is_goal=is_goal,
        statsbomb_xg=float(shot.get("statsbomb_xg", 0.0) or 0.0),
        match_id=match_id,
        event_id=str(ev.get("id", "")),
        minute=int(ev.get("minute", 0) or 0),
        period=int(ev.get("period", 1) or 1),
        player_name=(ev.get("player") or {}).get("name", ""),
        team_name=(ev.get("team") or {}).get("name", ""),
        player_id=int((ev.get("player") or {}).get("id", 0) or 0),
    )


# ── Style-profile event stream ──────────────────────────────────────────────
#
# The shot parser above answers "how good was that shot"; style metrics ask
# "how did the team play". They need the same event categories on the corpus
# side that the app stores on its own side, in Kawkab meters and
# possession-relative frames (each team attacking toward x = 105).

SB_STYLE_TYPES = {
    "Pass": "pass",
    "Carry": "carry",
    "Pressure": "pressure",
    "Interception": "interception",
    "Duel": "duel",
    "Ball Recovery": "recovery",
    "Clearance": "clearance",
    "Shot": "shot",
    "Foul Committed": "foul",
    # These two keep the corpus vocabulary identical to what the importer
    # stores, so PPDA/turnover denominators count the same events on both
    # sides ("50/50" is imported as a duel, bad behaviour as a foul).
    "50/50": "duel",
    "Bad Behaviour": "foul",
}


def _meters_or_none(loc: Any) -> tuple[float | None, float | None]:
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return None, None
    try:
        x_m, y_m = sb_to_meters(loc[:2])
    except (TypeError, ValueError):
        return None, None
    return float(x_m), float(y_m)


def normalize_statsbomb_events(raw: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per-team normalized event stream for style metrics.

    Mirrors the shape the stored-event adapter produces in
    ``services/elite_benchmark_service.py``: type, start/end position in
    Kawkab meters, pass completion, duel outcome. Events without a team
    or without a mapped type are dropped, never guessed.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for ev in raw:
        if not isinstance(ev, dict):
            continue
        type_name = (ev.get("type") or {}).get("name", "")
        event_type = SB_STYLE_TYPES.get(type_name)
        if event_type is None:
            continue
        team = (ev.get("team") or {}).get("name") or ""
        if not team:
            continue

        x, y = _meters_or_none(ev.get("location"))
        end_x = end_y = None
        completed: bool | None = None
        won: bool | None = None
        is_goal: bool | None = None

        if event_type == "pass":
            pass_data = ev.get("pass") or {}
            end_x, end_y = _meters_or_none(pass_data.get("end_location"))
            completed = pass_data.get("outcome") is None
        elif event_type == "carry":
            end_x, end_y = _meters_or_none((ev.get("carry") or {}).get("end_location"))
        elif event_type == "interception":
            outcome = ((ev.get("interception") or {}).get("outcome") or {}).get("name", "Won")
            won = outcome == "Won"
        elif event_type == "duel":
            duel_data = ev.get("50_50") if type_name == "50/50" else ev.get("duel")
            outcome = ((duel_data or {}).get("outcome") or {}).get("name", "")
            won = outcome == "Won"
        elif event_type == "shot":
            is_goal = ((ev.get("shot") or {}).get("outcome") or {}).get("name") == "Goal"

        player = ev.get("player") or {}
        out.setdefault(team, []).append(
            {
                "type": event_type,
                "x": x,
                "y": y,
                "end_x": end_x,
                "end_y": end_y,
                "completed": completed,
                "won": won,
                "is_goal": is_goal,
                "player_id": int(player.get("id", 0) or 0) or None,
                "player_name": player.get("name") or "",
            }
        )
    return out


def _nearest_opponent_distance(
    freeze_frame: list[dict[str, Any]] | None,
    shot_loc: list[float],
) -> float:
    """Distance from the shot position to the nearest opposing player (SB units)."""
    if not freeze_frame:
        return 999.0
    best = 999.0
    for p in freeze_frame:
        if p.get("teammate", False):
            continue
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue
        d = math.hypot(loc[0] - shot_loc[0], loc[1] - shot_loc[1])
        best = min(best, d)
    return best


def load_statsbomb_corpus(
    corpus_dir: str | Path,
    *,
    limit: int | None = None,
    progress: bool = False,
) -> Iterator[StatsBombMatch]:
    """Yield parsed matches from every ``*.json`` in the corpus directory.

    Non-list JSON files (competitions index, lineups, etc.) are skipped
    with a debug log, not an error.
    """
    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    for path in files:
        try:
            m = load_statsbomb_match(path)
            if m.n_events > 0 or m.shots:
                yield m
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("skipping %s: %s", path.name, exc)
            if progress:
                print(f"  [skip] {path.name}: {exc}")


def extract_shots_for_fitting(
    matches: list[StatsBombMatch] | Iterator[StatsBombMatch],
) -> tuple[list[StatsBombShot], int]:
    """Flatten matches into (shots, n_matches) for model fitting.

    Penalties are excluded from fitting (fixed 0.76 xG, not learned), and
    shots missing a location are dropped — both counts are logged so the
    training provenance is auditable.
    """
    shots: list[StatsBombShot] = []
    n_matches = 0
    n_penalties = 0
    for m in matches:
        n_matches += 1
        for s in m.shots:
            if s.shot_type == "penalty":
                n_penalties += 1
                continue
            shots.append(s)
    if n_penalties:
        logger.info("excluded %d penalties from xG fitting (fixed 0.76)", n_penalties)
    return shots, n_matches
