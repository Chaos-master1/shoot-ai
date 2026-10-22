"""Tests for TacticalWhiteboard — annotation layer, formations, SVG export, save/load."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs

install_kawkab_stubs()

import pytest

from kawkab.analysis.tactical_whiteboard import (
    FORMATION_TEMPLATES,
    Annotation,
    TacticalWhiteboard,
    WhiteboardState,
)


@pytest.fixture
def tw():
    return TacticalWhiteboard()


class TestTemplates:
    def test_list_templates(self, tw):
        templates = tw.list_templates()
        assert len(templates) >= 6
        assert any(t["name"] == "4-4-2" for t in templates)
        assert any(t["name"] == "4-3-3" for t in templates)

    def test_get_template(self, tw):
        pos = tw.get_template("4-4-2")
        assert pos is not None
        assert len(pos) == 11
        assert pos[0]["pos"] == "GK"

    def test_get_template_unknown(self, tw):
        assert tw.get_template("invalid") is None

    def test_template_counts(self, tw):
        for name in FORMATION_TEMPLATES:
            t = FORMATION_TEMPLATES[name]
            assert len(t.positions) == 11


class TestState:
    def test_create_state_empty(self, tw):
        state = tw.create_state("Test")
        assert state.name == "Test"
        assert len(state.players_home) == 0
        assert state.annotations == []

    def test_create_state_with_formation(self, tw):
        state = tw.create_state("Match", "4-3-3")
        assert len(state.players_home) == 11
        assert state.formation_home == "4-3-3"

    def test_get_state(self, tw):
        s1 = tw.create_state("A")
        s2 = tw.get_state(s1.id)
        assert s2 is not None
        assert s2.name == "A"

    def test_get_state_not_found(self, tw):
        assert tw.get_state("nonexistent") is None

    def test_list_states(self, tw):
        tw.create_state("A")
        tw.create_state("B")
        states = tw.list_states()
        assert len(states) == 2

    def test_delete_state(self, tw):
        s = tw.create_state("X")
        assert tw.delete_state(s.id) is True
        assert tw.get_state(s.id) is None

    def test_delete_state_not_found(self, tw):
        assert tw.delete_state("nonexistent") is False

    def test_update_state_name(self, tw):
        s = tw.create_state("Old")
        tw.update_state(s.id, {"name": "New"})
        assert tw.get_state(s.id).name == "New"

    def test_update_state_annotations(self, tw):
        s = tw.create_state("A")
        annos = [{"id": "1", "type": "arrow", "points": []}]
        tw.update_state(s.id, {"annotations": annos})
        assert len(tw.get_state(s.id).annotations) == 1

    def test_update_state_not_found(self, tw):
        assert tw.update_state("nonexistent", {}) is None


class TestAnnotations:
    def test_add_annotation(self, tw):
        s = tw.create_state("A")
        ann = tw.add_annotation(
            s.id,
            {
                "type": "arrow",
                "points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}],
                "color": "#ff0000",
            },
        )
        assert ann is not None
        assert ann.type == "arrow"

    def test_add_annotation_no_state(self, tw):
        assert tw.add_annotation("x", {}) is None

    def test_remove_annotation(self, tw):
        s = tw.create_state("A")
        ann = tw.add_annotation(s.id, {"type": "line", "points": []})
        assert tw.remove_annotation(s.id, ann.id) is True
        assert len(s.annotations) == 0

    def test_remove_annotation_not_found(self, tw):
        s = tw.create_state("A")
        assert tw.remove_annotation(s.id, "nonexistent") is False

    def test_clear_annotations(self, tw):
        s = tw.create_state("A")
        tw.add_annotation(s.id, {"type": "circle", "points": [{"x": 50, "y": 50}]})
        tw.add_annotation(s.id, {"type": "text", "points": [{"x": 30, "y": 30}], "label": "Test"})
        assert tw.clear_annotations(s.id) is True
        assert len(s.annotations) == 0


class TestPlayers:
    def test_set_players_from_formation(self, tw):
        s = tw.create_state("A")
        players = tw.set_players_from_formation(s.id, "4-2-3-1", "home")
        assert players is not None
        assert len(players) == 11
        assert s.formation_home == "4-2-3-1"

    def test_set_players_invalid_team(self, tw):
        s = tw.create_state("A")
        players = tw.set_players_from_formation(s.id, "4-4-2", "away")
        assert players is not None
        assert s.formation_away == "4-4-2"
        assert len(s.players_away) == 11

    def test_set_players_unknown_formation(self, tw):
        s = tw.create_state("A")
        assert tw.set_players_from_formation(s.id, "invalid") is None

    def test_move_player(self, tw):
        s = tw.create_state("A", "3-5-2")
        assert tw.move_player(s.id, 1, 20.0, 30.0, "home") is True
        assert s.players_home[1]["x"] == 20.0
        assert s.players_home[1]["y"] == 30.0

    def test_move_player_invalid_index(self, tw):
        s = tw.create_state("A", "4-4-2")
        assert tw.move_player(s.id, 99, 0, 0) is False

    def test_move_player_no_state(self, tw):
        assert tw.move_player("nonexistent", 0, 0, 0) is False


class TestSVG:
    def test_generate_svg_no_state(self, tw):
        assert tw.generate_svg("nonexistent") is None

    def test_generate_svg_empty(self, tw):
        s = tw.create_state("Empty")
        svg = tw.generate_svg(s.id)
        assert svg is not None
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "1a5c2a" in svg

    def test_generate_svg_with_players(self, tw):
        s = tw.create_state("WithPlayers", "4-4-2")
        svg = tw.generate_svg(s.id)
        assert svg is not None
        assert 'fill="#e74c3c"' in svg
        assert "1" in svg

    def test_generate_svg_with_away(self, tw):
        s = tw.create_state("Both")
        tw.set_players_from_formation(s.id, "4-3-3", "home")
        tw.set_players_from_formation(s.id, "3-5-2", "away")
        svg = tw.generate_svg(s.id)
        assert 'fill="#3498db"' in svg
        assert 'fill="#e74c3c"' in svg

    def test_generate_svg_with_annotations(self, tw):
        s = tw.create_state("Annotated")
        tw.add_annotation(
            s.id,
            {
                "type": "arrow",
                "points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}],
            },
        )
        tw.add_annotation(
            s.id,
            {
                "type": "circle",
                "points": [{"x": 50, "y": 50, "radius": 10}],
            },
        )
        svg = tw.generate_svg(s.id)
        assert "polygon" in svg or "line" in svg
        assert "circle" in svg

    def test_generate_svg_with_ball(self, tw):
        s = tw.create_state("Ball")
        tw.update_state(s.id, {"ball_position": {"x": 50, "y": 50}})
        svg = tw.generate_svg(s.id)
        assert 'fill="white"' in svg


class TestAnnotationHelpers:
    def test_generate_player_run(self, tw):
        ann = tw.generate_player_run(10, 10, 50, 50, label="LB -> LW")
        assert ann.type == "arrow"
        assert ann.label == "LB -> LW"
        assert len(ann.points) == 2

    def test_generate_pass(self, tw):
        ann = tw.generate_pass(20, 30, 60, 40)
        assert ann.type == "arrow"
        assert ann.dashed is True

    def test_generate_shot(self, tw):
        ann = tw.generate_shot(70, 50, 100, 50)
        assert ann.type == "arrow"
        assert ann.color == "#e74c3c"


class TestSerialization:
    def test_annotation_dataclass(self):
        ann = Annotation(type="circle", points=[{"x": 50, "y": 50}], label="Center")
        d = ann.to_dict()
        assert d["type"] == "circle"
        assert d["label"] == "Center"
        assert len(d["id"]) == 12

    def test_state_dataclass(self):
        state = WhiteboardState(name="My Board", formation_home="4-4-2")
        d = state.to_dict()
        assert d["name"] == "My Board"
        assert d["formation_home"] == "4-4-2"
        assert "id" in d
        assert "timestamp" in d

    def test_to_json(self, tw):
        s = tw.create_state("JSON Test", "4-3-3")
        result = tw.to_json(s.id)
        assert result is not None
        import json

        data = json.loads(result)
        assert data["name"] == "JSON Test"
        assert data["formation_home"] == "4-3-3"

    def test_to_json_no_state(self, tw):
        assert tw.to_json("nonexistent") is None


class TestEdgeCases:
    def test_empty_state_list(self, tw):
        assert tw.list_states() == []

    def test_state_id_uniqueness(self, tw):
        s1 = tw.create_state("A")
        s2 = tw.create_state("B")
        assert s1.id != s2.id

    def test_generate_svg_line_annotation(self, tw):
        s = tw.create_state("Lines")
        tw.add_annotation(
            s.id,
            {
                "type": "line",
                "points": [{"x": 0, "y": 0}, {"x": 100, "y": 100}],
            },
        )
        svg = tw.generate_svg(s.id)
        assert "line" in svg

    def test_generate_svg_text_annotation(self, tw):
        s = tw.create_state("Text")
        tw.add_annotation(
            s.id,
            {
                "type": "text",
                "points": [{"x": 50, "y": 30}],
                "label": "Press Here",
                "color": "#ffff00",
            },
        )
        svg = tw.generate_svg(s.id)
        assert "Press Here" in svg

    def test_generate_svg_zone_annotation(self, tw):
        s = tw.create_state("Zone")
        tw.add_annotation(
            s.id,
            {
                "type": "zone",
                "points": [
                    {"x": 20, "y": 20},
                    {"x": 40, "y": 20},
                    {"x": 40, "y": 40},
                    {"x": 20, "y": 40},
                ],
                "color": "#ff0000",
                "opacity": 0.3,
            },
        )
        svg = tw.generate_svg(s.id)
        assert "Z" in svg

    def test_custom_width_height_svg(self, tw):
        s = tw.create_state("Custom")
        svg = tw.generate_svg(s.id, width=800, height=500)
        assert 'viewBox="0 0 800 500"' in svg

    def test_update_state_pitch_orientation(self, tw):
        s = tw.create_state("Orientation")
        tw.update_state(s.id, {"pitch_orientation": "vertical"})
        assert tw.get_state(s.id).pitch_orientation == "vertical"
