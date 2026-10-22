"""Typed event model for football match events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from kawkab.core.logging import get_logger


class EventType(Enum):
    GOAL = "goal"
    SHOT = "shot"
    PASS = "pass"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    DRIBBLE = "dribble"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    THROW_IN = "throw_in"
    CLEARANCE = "clearance"
    CROSS = "cross"
    BLOCK = "block"
    CARRY = "carry"
    DUEL = "duel"
    FOUL = "foul"
    OFFSIDE = "offside"
    HAND_BALL = "hand_ball"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    GOAL_KICK = "goal_kick"
    PENALTY = "penalty"
    BALL_OUT = "ball_out"
    OUT_OF_PLAY = "out_of_play"
    SAVE = "save"


class BodyPart(Enum):
    RIGHT_FOOT = "right_foot"
    LEFT_FOOT = "left_foot"
    HEAD = "head"
    OTHER = "other"


class PassType(Enum):
    STANDARD = "standard"
    THROUGH_BALL = "through_ball"
    CROSS = "cross"
    SWITCH = "switch"
    HEADER = "header"
    BACK_PASS = "back_pass"
    ONE_TOUCH = "one_touch"
    FLICK_ON = "flick_on"
    LONG_BALL = "long_ball"


class ShotType(Enum):
    OPEN_PLAY = "open_play"
    FREE_KICK = "free_kick"
    PENALTY = "penalty"
    HEADER = "header"
    VOLLEY = "volley"
    HALF_VOLLEY = "half_volley"


class AssistType(Enum):
    STANDARD = "standard"
    THROUGH_BALL = "through_ball"
    CROSS = "cross"
    PULL_BACK = "pull_back"
    HEAD_PASS = "head_pass"
    CUTBACK = "cutback"
    UNKNOWN = "unknown"


class TackleType(Enum):
    STANDING = "standing"
    SLIDING = "sliding"
    SHOULDER = "shoulder"


@dataclass
class PressureContext:
    nearest_defender_distance: float = 0.0
    nearest_defender_angle: float = 0.0
    defenders_within_5m: int = 0
    is_pressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dist_def": round(self.nearest_defender_distance, 2),
            "angle_def": round(self.nearest_defender_angle, 1),
            "n_def_5m": self.defenders_within_5m,
            "pressed": self.is_pressed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PressureContext | None:
        if d is None:
            return None
        return cls(
            nearest_defender_distance=d.get("dist_def", 0.0),
            nearest_defender_angle=d.get("angle_def", 0.0),
            defenders_within_5m=d.get("n_def_5m", 0),
            is_pressed=d.get("pressed", False),
        )


@dataclass
class BaseEvent:
    type: EventType = EventType.PASS
    timestamp: float = 0.0
    team: str = "unknown"
    track_id: int | None = None
    x: float | None = None
    y: float | None = None
    confidence: float = 1.0
    pressure: PressureContext | None = None
    period: int = 1

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "team": self.team,
            "track_id": self.track_id,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "period": self.period,
        }
        if self.pressure is not None:
            d["pressure"] = self.pressure.to_dict()
        return d

    @classmethod
    def _from_base_dict(cls, d: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timestamp": d.get("timestamp", 0.0),
            "team": d.get("team", "unknown"),
            "track_id": d.get("track_id"),
            "x": d.get("x"),
            "y": d.get("y"),
            "confidence": d.get("confidence", 1.0),
            "period": d.get("period", 1),
            "pressure": PressureContext.from_dict(d.get("pressure")),
        }
        return kwargs


@dataclass
class PassEvent(BaseEvent):
    pass_type: PassType = PassType.STANDARD
    to_track_id: int | None = None
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    completed: bool = True
    length_m: float = 0.0
    speed_mps: float = 0.0
    body_part: BodyPart = BodyPart.RIGHT_FOOT
    assist_type: AssistType = AssistType.STANDARD
    is_through_ball: bool = False
    is_cross: bool = False
    is_switch: bool = False
    is_key_pass: bool = False
    is_assist: bool = False
    is_progressive: bool = False

    def __post_init__(self) -> None:
        self.type = EventType.PASS

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "pass_type": self.pass_type.value,
                "to_track_id": self.to_track_id,
                "start_x": self.start_x,
                "start_y": self.start_y,
                "end_x": self.end_x,
                "end_y": self.end_y,
                "completed": self.completed,
                "length_m": round(self.length_m, 1),
                "speed_mps": round(self.speed_mps, 2),
                "body_part": self.body_part.value,
                "is_through_ball": self.is_through_ball,
                "is_cross": self.is_cross,
                "is_switch": self.is_switch,
                "is_key_pass": self.is_key_pass,
                "is_assist": self.is_assist,
                "is_progressive": self.is_progressive,
                "assist_type": self.assist_type.value,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PassEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["pass_type"] = _parse_enum(d, "pass_type", PassType, PassType.STANDARD)
        kwargs["body_part"] = _parse_enum(d, "body_part", BodyPart, BodyPart.RIGHT_FOOT)
        kwargs["assist_type"] = _parse_enum(d, "assist_type", AssistType, AssistType.STANDARD)
        for field_name in ("to_track_id", "start_x", "start_y", "end_x", "end_y"):
            kwargs[field_name] = d.get(field_name)
        kwargs["completed"] = d.get("completed", True)
        kwargs["length_m"] = float(d.get("length_m", 0))
        kwargs["speed_mps"] = float(d.get("speed_mps", 0))
        for bool_field in (
            "is_through_ball",
            "is_cross",
            "is_switch",
            "is_key_pass",
            "is_assist",
            "is_progressive",
        ):
            kwargs[bool_field] = d.get(bool_field, False)
        return cls(**kwargs)


@dataclass
class ShotEvent(BaseEvent):
    on_target: bool = False
    # Honest-absent: None means the position was never known (no
    # homography / no metadata). Consumers fall back via `or` — a
    # fabricated 0.0 would silently poison xG feature vectors.
    distance_m: float | None = None
    angle_deg: float | None = None
    body_part: BodyPart = BodyPart.RIGHT_FOOT
    shot_type: ShotType = ShotType.OPEN_PLAY
    is_one_on_one: bool = False
    is_volley: bool = False
    xg: float = 0.0
    psxg: float = 0.0
    gk_position_x: float | None = None
    gk_position_y: float | None = None
    was_pressed: bool = False
    previous_action: str | None = None

    def __post_init__(self) -> None:
        self.type = EventType.SHOT

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "on_target": self.on_target,
                "distance_m": round(self.distance_m, 1) if self.distance_m is not None else None,
                "angle_deg": round(self.angle_deg, 1) if self.angle_deg is not None else None,
                "body_part": self.body_part.value,
                "shot_type": self.shot_type.value,
                "is_one_on_one": self.is_one_on_one,
                "is_volley": self.is_volley,
                "xg": round(self.xg, 4),
                "psxg": round(self.psxg, 4),
                "gk_position_x": self.gk_position_x,
                "gk_position_y": self.gk_position_y,
                "was_pressed": self.was_pressed,
                "previous_action": self.previous_action,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShotEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["body_part"] = _parse_enum(d, "body_part", BodyPart, BodyPart.RIGHT_FOOT)
        kwargs["shot_type"] = _parse_enum(d, "shot_type", ShotType, ShotType.OPEN_PLAY)
        kwargs["on_target"] = d.get("on_target", False)
        kwargs["distance_m"] = float(d["distance_m"]) if d.get("distance_m") is not None else None
        kwargs["angle_deg"] = float(d["angle_deg"]) if d.get("angle_deg") is not None else None
        kwargs["is_one_on_one"] = d.get("is_one_on_one", False)
        kwargs["is_volley"] = d.get("is_volley", False)
        kwargs["xg"] = float(d.get("xg", 0))
        kwargs["psxg"] = float(d.get("psxg", 0))
        kwargs["gk_position_x"] = d.get("gk_position_x")
        kwargs["gk_position_y"] = d.get("gk_position_y")
        kwargs["was_pressed"] = d.get("was_pressed", False)
        kwargs["previous_action"] = d.get("previous_action")
        return cls(**kwargs)


@dataclass
class CarryEvent(BaseEvent):
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    distance_m: float = 0.0
    is_progressive: bool = False
    end_zone_threat: float = 0.0
    direction_change_deg: float = 0.0
    body_part: BodyPart = BodyPart.RIGHT_FOOT

    def __post_init__(self) -> None:
        self.type = EventType.CARRY

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "start_x": self.start_x,
                "start_y": self.start_y,
                "end_x": self.end_x,
                "end_y": self.end_y,
                "distance_m": round(self.distance_m, 1),
                "is_progressive": self.is_progressive,
                "end_zone_threat": round(self.end_zone_threat, 4),
                "direction_change_deg": round(self.direction_change_deg, 1),
                "body_part": self.body_part.value,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CarryEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["body_part"] = _parse_enum(d, "body_part", BodyPart, BodyPart.RIGHT_FOOT)
        for f in ("start_x", "start_y", "end_x", "end_y"):
            kwargs[f] = d.get(f)
        kwargs["distance_m"] = float(d.get("distance_m", 0))
        kwargs["is_progressive"] = d.get("is_progressive", False)
        kwargs["end_zone_threat"] = float(d.get("end_zone_threat", 0))
        kwargs["direction_change_deg"] = float(d.get("direction_change_deg", 0))
        return cls(**kwargs)


@dataclass
class TackleEvent(BaseEvent):
    tackle_type: TackleType = TackleType.STANDING
    succeeded: bool = True
    opponent_track_id: int | None = None

    def __post_init__(self) -> None:
        self.type = EventType.TACKLE

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "tackle_type": self.tackle_type.value,
                "succeeded": self.succeeded,
                "opponent_track_id": self.opponent_track_id,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TackleEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["tackle_type"] = _parse_enum(d, "tackle_type", TackleType, TackleType.STANDING)
        kwargs["succeeded"] = d.get("succeeded", True)
        kwargs["opponent_track_id"] = d.get("opponent_track_id")
        return cls(**kwargs)


@dataclass
class InterceptionEvent(BaseEvent):
    opponent_pass_from: int | None = None
    opponent_pass_to: int | None = None
    led_to_attack: bool = False

    def __post_init__(self) -> None:
        self.type = EventType.INTERCEPTION

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "opponent_pass_from": self.opponent_pass_from,
                "opponent_pass_to": self.opponent_pass_to,
                "led_to_attack": self.led_to_attack,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterceptionEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["opponent_pass_from"] = d.get("opponent_pass_from")
        kwargs["opponent_pass_to"] = d.get("opponent_pass_to")
        kwargs["led_to_attack"] = d.get("led_to_attack", False)
        return cls(**kwargs)


@dataclass
class GoalEvent(BaseEvent):
    shot_xg: float = 0.0
    assist_player_id: int | None = None
    body_part: BodyPart = BodyPart.RIGHT_FOOT

    def __post_init__(self) -> None:
        self.type = EventType.GOAL

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "shot_xg": round(self.shot_xg, 4),
                "assist_player_id": self.assist_player_id,
                "body_part": self.body_part.value,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoalEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["shot_xg"] = float(d.get("shot_xg", 0))
        kwargs["assist_player_id"] = d.get("assist_player_id")
        kwargs["body_part"] = _parse_enum(d, "body_part", BodyPart, BodyPart.RIGHT_FOOT)
        return cls(**kwargs)


@dataclass
class FoulEvent(BaseEvent):
    foul_type: str = ""
    card_color: str | None = None

    def __post_init__(self) -> None:
        self.type = EventType.FOUL

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "foul_type": self.foul_type,
                "card_color": self.card_color,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FoulEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["foul_type"] = d.get("foul_type", "")
        kwargs["card_color"] = d.get("card_color")
        return cls(**kwargs)


@dataclass
class CornerEvent(BaseEvent):
    delivery_zone: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.CORNER

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({"delivery_zone": self.delivery_zone})
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CornerEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["delivery_zone"] = d.get("delivery_zone", "")
        return cls(**kwargs)


@dataclass
class FreeKickEvent(BaseEvent):
    kick_type: str = ""
    wall_size: int = 0

    def __post_init__(self) -> None:
        self.type = EventType.FREE_KICK

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "kick_type": self.kick_type,
                "wall_size": self.wall_size,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FreeKickEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["kick_type"] = d.get("kick_type", "")
        kwargs["wall_size"] = int(d.get("wall_size", 0))
        return cls(**kwargs)


@dataclass
class OffsideEvent(BaseEvent):
    offside_type: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.OFFSIDE

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({"offside_type": self.offside_type})
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OffsideEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["offside_type"] = d.get("offside_type", "")
        return cls(**kwargs)


@dataclass
class SubstitutionEvent(BaseEvent):
    player_off: int | None = None
    player_on: int | None = None

    def __post_init__(self) -> None:
        self.type = EventType.SUBSTITUTION

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "player_off": self.player_off,
                "player_on": self.player_on,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubstitutionEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["player_off"] = d.get("player_off")
        kwargs["player_on"] = d.get("player_on")
        return cls(**kwargs)


@dataclass
class SaveEvent(BaseEvent):
    shot_xg: float = 0.0
    save_type: str = ""
    rebound: bool = False

    def __post_init__(self) -> None:
        self.type = EventType.SAVE

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "shot_xg": round(self.shot_xg, 4),
                "save_type": self.save_type,
                "rebound": self.rebound,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SaveEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["shot_xg"] = float(d.get("shot_xg", 0))
        kwargs["save_type"] = d.get("save_type", "")
        kwargs["rebound"] = d.get("rebound", False)
        return cls(**kwargs)


@dataclass
class CardEvent(BaseEvent):
    card_color: str = ""
    card_reason: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.YELLOW_CARD

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "card_color": self.card_color,
                "card_reason": self.card_reason,
            }
        )
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CardEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["card_color"] = d.get("card_color", "")
        kwargs["card_reason"] = d.get("card_reason", "")
        obj = cls(**kwargs)
        type_str = d.get("type", "yellow_card")
        obj.type = EventType.RED_CARD if type_str == "red_card" else EventType.YELLOW_CARD
        return obj


@dataclass
class BallOutEvent(BaseEvent):
    out_type: str = ""

    def __post_init__(self) -> None:
        self.type = EventType.BALL_OUT

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({"out_type": self.out_type})
        return base

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BallOutEvent:
        kwargs = cls._from_base_dict(d)
        kwargs["out_type"] = d.get("out_type", "")
        return cls(**kwargs)


_EVENT_CLASSES: dict[EventType, type[BaseEvent]] = {
    EventType.GOAL: GoalEvent,
    EventType.SHOT: ShotEvent,
    EventType.PASS: PassEvent,
    EventType.TACKLE: TackleEvent,
    EventType.INTERCEPTION: InterceptionEvent,
    EventType.CARRY: CarryEvent,
    EventType.FOUL: FoulEvent,
    EventType.CORNER: CornerEvent,
    EventType.FREE_KICK: FreeKickEvent,
    EventType.OFFSIDE: OffsideEvent,
    EventType.SUBSTITUTION: SubstitutionEvent,
    EventType.SAVE: SaveEvent,
    EventType.BALL_OUT: BallOutEvent,
    EventType.YELLOW_CARD: CardEvent,
    EventType.RED_CARD: CardEvent,
}


def event_from_dict(d: dict[str, Any]) -> BaseEvent:
    type_str = d.get("type", "pass")
    try:
        etype = EventType(type_str)
    except ValueError:
        etype = EventType.PASS

    from kawkab.core.coordinate_validator import CoordinateValidator

    result = CoordinateValidator.validate_event_spatial(d)
    if not result.valid:
        raise ValueError(f"Invalid spatial coordinates in event: {result.errors}")
    if result.clamped:
        logger = get_logger(__name__)
        logger.warning(f"Spatial coordinates clamped for event type={type_str}: {result.warnings}")

    # Every concrete event class defines from_dict; the shared base class
    # deliberately does not (its from_dict helpers live per-subclass).
    cls: type[PassEvent] = _EVENT_CLASSES.get(etype, PassEvent)  # type: ignore[assignment]
    return cls.from_dict(d)


def _parse_enum(d: dict[str, Any], key: str, enum_cls: type, default: Any) -> Any:
    val = d.get(key)
    if val is None:
        return default
    try:
        return enum_cls(val)
    except (ValueError, TypeError):
        return default
