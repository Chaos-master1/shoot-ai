"""Tests for the typed event model."""

from kawkab.core.events import (
    BaseEvent,
    BodyPart,
    CarryEvent,
    EventType,
    InterceptionEvent,
    PassEvent,
    PassType,
    PressureContext,
    ShotEvent,
    ShotType,
    TackleEvent,
    TackleType,
    event_from_dict,
)


class TestEventTypes:
    def test_pass_event_defaults(self):
        e = PassEvent(timestamp=10.0, team="home", track_id=1, to_track_id=2)
        assert e.type == EventType.PASS
        assert e.timestamp == 10.0
        assert e.team == "home"
        assert e.track_id == 1
        assert e.to_track_id == 2
        assert e.completed is True
        assert e.pass_type == PassType.STANDARD
        assert e.body_part == BodyPart.RIGHT_FOOT
        assert e.is_through_ball is False

    def test_shot_event_defaults(self):
        e = ShotEvent(timestamp=15.0, team="away", track_id=3, xg=0.25)
        assert e.type == EventType.SHOT
        assert e.xg == 0.25
        assert e.on_target is False
        assert e.shot_type == ShotType.OPEN_PLAY

    def test_carry_event(self):
        e = CarryEvent(timestamp=5.0, team="home", track_id=1, distance_m=12.5, is_progressive=True)
        assert e.type == EventType.CARRY
        assert e.distance_m == 12.5
        assert e.is_progressive is True

    def test_tackle_event(self):
        e = TackleEvent(timestamp=20.0, team="home", track_id=4, tackle_type=TackleType.SLIDING)
        assert e.type == EventType.TACKLE
        assert e.tackle_type == TackleType.SLIDING
        assert e.succeeded is True

    def test_interception_event(self):
        e = InterceptionEvent(timestamp=25.0, team="away", track_id=5, led_to_attack=True)
        assert e.type == EventType.INTERCEPTION
        assert e.led_to_attack is True

    def test_base_event_no_pressure(self):
        e = BaseEvent(timestamp=0.0, team="home")
        assert e.pressure is None
        d = e.to_dict()
        assert "pressure" not in d

    def test_base_event_with_pressure(self):
        e = BaseEvent(
            timestamp=0.0,
            team="home",
            pressure=PressureContext(nearest_defender_distance=1.5, is_pressed=True),
        )
        assert e.pressure is not None
        assert e.pressure.nearest_defender_distance == 1.5
        assert e.pressure.is_pressed is True

    def test_shot_event_pressure_flag(self):
        e = ShotEvent(timestamp=10.0, team="home", was_pressed=True)
        assert e.was_pressed is True


class TestSerialization:
    def test_pass_to_dict_roundtrip(self):
        e = PassEvent(
            timestamp=10.0,
            team="home",
            track_id=1,
            to_track_id=2,
            pass_type=PassType.THROUGH_BALL,
            length_m=18.5,
            is_progressive=True,
            body_part=BodyPart.LEFT_FOOT,
        )
        d = e.to_dict()
        restored = PassEvent.from_dict(d)
        assert restored.timestamp == e.timestamp
        assert restored.team == e.team
        assert restored.track_id == e.track_id
        assert restored.to_track_id == e.to_track_id
        assert restored.pass_type == PassType.THROUGH_BALL
        assert restored.length_m == e.length_m
        assert restored.is_progressive is True
        assert restored.body_part == BodyPart.LEFT_FOOT

    def test_shot_to_dict_roundtrip(self):
        e = ShotEvent(
            timestamp=15.0,
            team="away",
            track_id=3,
            on_target=True,
            distance_m=12.3,
            xg=0.142,
            body_part=BodyPart.HEAD,
            shot_type=ShotType.VOLLEY,
            was_pressed=True,
        )
        d = e.to_dict()
        restored = ShotEvent.from_dict(d)
        assert restored.on_target is True
        assert restored.distance_m == e.distance_m
        assert restored.xg == e.xg
        assert restored.body_part == BodyPart.HEAD
        assert restored.shot_type == ShotType.VOLLEY
        assert restored.was_pressed is True

    def test_event_from_dict_dispatch_pass(self):
        d = {"type": "pass", "timestamp": 10.0, "team": "home", "track_id": 1, "to_track_id": 2}
        e = event_from_dict(d)
        assert isinstance(e, PassEvent)
        assert e.track_id == 1
        assert e.to_track_id == 2

    def test_event_from_dict_dispatch_shot(self):
        d = {"type": "shot", "timestamp": 15.0, "team": "away", "track_id": 3, "xg": 0.25}
        e = event_from_dict(d)
        assert isinstance(e, ShotEvent)
        assert e.xg == 0.25

    def test_event_from_dict_with_pressure(self):
        d = {
            "type": "pass",
            "timestamp": 10.0,
            "team": "home",
            "track_id": 1,
            "pressure": {"dist_def": 1.2, "angle_def": 15.0, "n_def_5m": 2, "pressed": True},
        }
        e = event_from_dict(d)
        assert isinstance(e, PassEvent)
        assert e.pressure is not None
        assert e.pressure.nearest_defender_distance == 1.2
        assert e.pressure.is_pressed is True
        assert e.pressure.defenders_within_5m == 2

    def test_event_from_dict_fallback_on_bad_type(self):
        d = {"type": "unknown_event_type", "timestamp": 0.0, "team": "home"}
        e = event_from_dict(d)
        assert isinstance(e, PassEvent)

    def test_pressure_context_to_dict(self):
        pc = PressureContext(nearest_defender_distance=2.5, is_pressed=True)
        d = pc.to_dict()
        assert d["dist_def"] == 2.5
        assert d["pressed"] is True

    def test_pressure_context_from_dict_none(self):
        assert PressureContext.from_dict(None) is None
