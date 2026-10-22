"""Tests for the events module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.core.events import (
    AssistType,
    BodyPart,
    EventType,
    PassType,
    PressureContext,
    ShotEvent,
    ShotType,
    TackleType,
    event_from_dict,
)


class TestEventType:
    """15 tests for EventType enum and event model."""

    def test_event_type_values(self):
        assert EventType.GOAL.value == "goal"
        assert EventType.SHOT.value == "shot"
        assert EventType.PASS.value == "pass"
        assert EventType.TACKLE.value == "tackle"
        assert EventType.DUEL.value == "duel"

    def test_event_type_members(self):
        assert len(EventType) >= 20
        assert EventType("goal") == EventType.GOAL
        assert EventType("shot") == EventType.SHOT

    def test_event_type_unique(self):
        vals = [e.value for e in EventType]
        assert len(vals) == len(set(vals))

    def test_body_part_values(self):
        assert BodyPart.RIGHT_FOOT.value == "right_foot"
        assert BodyPart.HEAD.value == "head"

    def test_pass_type_values(self):
        assert PassType.THROUGH_BALL.value == "through_ball"
        assert PassType.CROSS.value == "cross"

    def test_shot_type_values(self):
        assert ShotType.PENALTY.value == "penalty"
        assert ShotType.FREE_KICK.value == "free_kick"

    def test_assist_type_values(self):
        assert AssistType.THROUGH_BALL.value == "through_ball"
        assert AssistType.CROSS.value == "cross"

    def test_tackle_type_values(self):
        assert TackleType.SLIDING.value == "sliding"
        assert TackleType.STANDING.value == "standing"

    def test_pressure_context_defaults(self):
        pc = PressureContext()
        assert pc.nearest_defender_distance == 0.0
        assert pc.defenders_within_5m == 0
        assert pc.is_pressed is False

    def test_pressure_context_to_dict(self):
        pc = PressureContext(nearest_defender_distance=2.5, is_pressed=True)
        d = pc.to_dict()
        assert d["dist_def"] == 2.5
        assert d["pressed"] is True

    def test_pressure_context_from_dict(self):
        d = {"dist_def": 1.5, "angle_def": 30.0, "n_def_5m": 2, "pressed": True}
        pc = PressureContext.from_dict(d)
        assert pc is not None
        assert pc.nearest_defender_distance == 1.5
        assert pc.is_pressed is True

    def test_pressure_context_from_dict_none(self):
        assert PressureContext.from_dict(None) is None

    def test_shot_event_defaults(self):
        se = ShotEvent()
        assert se.xg == 0.0
        assert se.on_target is False
        assert se.shot_type == ShotType.OPEN_PLAY

    def test_shot_event_to_dict(self):
        se = ShotEvent(xg=0.45, on_target=True)
        d = se.to_dict()
        assert d["xg"] == 0.45
        assert d["on_target"] is True

    def test_event_from_dict_shot(self):
        d = {"type": "shot", "xg": 0.5, "on_target": True}
        ev = event_from_dict(d)
        assert ev is not None
        assert hasattr(ev, "xg")
        assert ev.xg == 0.5
