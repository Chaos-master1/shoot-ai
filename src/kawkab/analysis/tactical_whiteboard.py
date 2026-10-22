"""Tactical whiteboard — annotation layer for coach markup on pitch diagrams."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class AnnotationType(Enum):
    ARROW = "arrow"
    LINE = "line"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    FREEHAND = "freehand"
    TEXT = "text"
    PLAYER = "player"
    BALL = "ball"
    CONE = "cone"
    ZONE = "zone"


@dataclass
class Point:
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": round(self.x, 1), "y": round(self.y, 1)}


@dataclass
class Annotation:
    id: str = ""
    type: str = "arrow"
    points: list[dict] = field(default_factory=list)
    color: str = "#ffffff"
    width: float = 3.0
    opacity: float = 1.0
    dashed: bool = False
    label: str = ""
    player_number: str = ""
    layer: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "points": self.points,
            "color": self.color,
            "width": self.width,
            "opacity": self.opacity,
            "dashed": self.dashed,
            "label": self.label,
            "player_number": self.player_number,
            "layer": self.layer,
            "created_at": self.created_at,
        }


@dataclass
class FormationTemplate:
    name: str
    label: str
    positions: list[dict] = field(default_factory=list)


@dataclass
class WhiteboardState:
    id: str = ""
    name: str = ""
    pitch_orientation: str = "horizontal"
    annotations: list[dict] = field(default_factory=list)
    players_home: list[dict] = field(default_factory=list)
    players_away: list[dict] = field(default_factory=list)
    ball_position: dict | None = None
    formation_home: str = ""
    formation_away: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "pitch_orientation": self.pitch_orientation,
            "annotations": self.annotations,
            "players_home": self.players_home,
            "players_away": self.players_away,
            "ball_position": self.ball_position,
            "formation_home": self.formation_home,
            "formation_away": self.formation_away,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# Standard formation templates (normalized 0-100 coordinates)
FORMATION_TEMPLATES: dict[str, FormationTemplate] = {
    "4-4-2": FormationTemplate(
        "4-4-2",
        "4-4-2",
        [
            {"x": 5, "y": 50, "pos": "GK", "num": 1},
            {"x": 15, "y": 15, "pos": "LB", "num": 3},
            {"x": 15, "y": 38, "pos": "CB", "num": 4},
            {"x": 15, "y": 62, "pos": "CB", "num": 5},
            {"x": 15, "y": 85, "pos": "RB", "num": 2},
            {"x": 30, "y": 15, "pos": "LM", "num": 11},
            {"x": 30, "y": 38, "pos": "CM", "num": 8},
            {"x": 30, "y": 62, "pos": "CM", "num": 6},
            {"x": 30, "y": 85, "pos": "RM", "num": 7},
            {"x": 50, "y": 35, "pos": "ST", "num": 9},
            {"x": 50, "y": 65, "pos": "ST", "num": 10},
        ],
    ),
    "4-3-3": FormationTemplate(
        "4-3-3",
        "4-3-3",
        [
            {"x": 5, "y": 50, "pos": "GK", "num": 1},
            {"x": 15, "y": 15, "pos": "LB", "num": 3},
            {"x": 15, "y": 38, "pos": "CB", "num": 4},
            {"x": 15, "y": 62, "pos": "CB", "num": 5},
            {"x": 15, "y": 85, "pos": "RB", "num": 2},
            {"x": 30, "y": 38, "pos": "CM", "num": 8},
            {"x": 30, "y": 50, "pos": "CDM", "num": 6},
            {"x": 30, "y": 62, "pos": "CM", "num": 10},
            {"x": 50, "y": 20, "pos": "LW", "num": 7},
            {"x": 55, "y": 50, "pos": "ST", "num": 9},
            {"x": 50, "y": 80, "pos": "RW", "num": 11},
        ],
    ),
    "3-5-2": FormationTemplate(
        "3-5-2",
        "3-5-2",
        [
            {"x": 5, "y": 50, "pos": "GK", "num": 1},
            {"x": 15, "y": 30, "pos": "CB", "num": 4},
            {"x": 15, "y": 50, "pos": "CB", "num": 5},
            {"x": 15, "y": 70, "pos": "CB", "num": 3},
            {"x": 30, "y": 10, "pos": "LWB", "num": 2},
            {"x": 30, "y": 30, "pos": "CM", "num": 8},
            {"x": 30, "y": 50, "pos": "CDM", "num": 6},
            {"x": 30, "y": 70, "pos": "CM", "num": 10},
            {"x": 30, "y": 90, "pos": "RWB", "num": 7},
            {"x": 50, "y": 35, "pos": "ST", "num": 9},
            {"x": 50, "y": 65, "pos": "ST", "num": 11},
        ],
    ),
    "4-2-3-1": FormationTemplate(
        "4-2-3-1",
        "4-2-3-1",
        [
            {"x": 5, "y": 50, "pos": "GK", "num": 1},
            {"x": 15, "y": 15, "pos": "LB", "num": 3},
            {"x": 15, "y": 38, "pos": "CB", "num": 4},
            {"x": 15, "y": 62, "pos": "CB", "num": 5},
            {"x": 15, "y": 85, "pos": "RB", "num": 2},
            {"x": 28, "y": 35, "pos": "CDM", "num": 6},
            {"x": 28, "y": 65, "pos": "CDM", "num": 8},
            {"x": 42, "y": 20, "pos": "LW", "num": 7},
            {"x": 42, "y": 50, "pos": "CAM", "num": 10},
            {"x": 42, "y": 80, "pos": "RW", "num": 11},
            {"x": 58, "y": 50, "pos": "ST", "num": 9},
        ],
    ),
    "3-4-3": FormationTemplate(
        "3-4-3",
        "3-4-3",
        [
            {"x": 5, "y": 50, "pos": "GK", "num": 1},
            {"x": 15, "y": 30, "pos": "CB", "num": 4},
            {"x": 15, "y": 50, "pos": "CB", "num": 5},
            {"x": 15, "y": 70, "pos": "CB", "num": 3},
            {"x": 30, "y": 10, "pos": "LM", "num": 2},
            {"x": 30, "y": 38, "pos": "CM", "num": 8},
            {"x": 30, "y": 62, "pos": "CM", "num": 6},
            {"x": 30, "y": 90, "pos": "RM", "num": 7},
            {"x": 50, "y": 20, "pos": "LW", "num": 11},
            {"x": 55, "y": 50, "pos": "ST", "num": 9},
            {"x": 50, "y": 80, "pos": "RW", "num": 10},
        ],
    ),
    "5-3-2": FormationTemplate(
        "5-3-2",
        "5-3-2",
        [
            {"x": 5, "y": 50, "pos": "GK", "num": 1},
            {"x": 12, "y": 10, "pos": "LWB", "num": 3},
            {"x": 12, "y": 30, "pos": "CB", "num": 4},
            {"x": 12, "y": 50, "pos": "CB", "num": 5},
            {"x": 12, "y": 70, "pos": "CB", "num": 6},
            {"x": 12, "y": 90, "pos": "RWB", "num": 2},
            {"x": 28, "y": 30, "pos": "CM", "num": 8},
            {"x": 28, "y": 50, "pos": "CDM", "num": 7},
            {"x": 28, "y": 70, "pos": "CM", "num": 10},
            {"x": 48, "y": 35, "pos": "ST", "num": 9},
            {"x": 48, "y": 65, "pos": "ST", "num": 11},
        ],
    ),
    "4-1-4-1": FormationTemplate(
        "4-1-4-1",
        "4-1-4-1",
        [
            {"x": 5, "y": 50, "pos": "GK", "num": 1},
            {"x": 15, "y": 15, "pos": "LB", "num": 3},
            {"x": 15, "y": 38, "pos": "CB", "num": 4},
            {"x": 15, "y": 62, "pos": "CB", "num": 5},
            {"x": 15, "y": 85, "pos": "RB", "num": 2},
            {"x": 25, "y": 50, "pos": "CDM", "num": 6},
            {"x": 38, "y": 15, "pos": "LM", "num": 7},
            {"x": 38, "y": 38, "pos": "CM", "num": 8},
            {"x": 38, "y": 62, "pos": "CM", "num": 10},
            {"x": 38, "y": 85, "pos": "RM", "num": 11},
            {"x": 55, "y": 50, "pos": "ST", "num": 9},
        ],
    ),
}


class TacticalWhiteboard:
    """Tactical whiteboard service for coach annotation and formation planning."""

    PITCH_W = 100.0
    PITCH_H = 100.0

    def __init__(self):
        self._states: dict[str, WhiteboardState] = {}
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def list_templates(self) -> list[dict]:
        return [
            {"name": t.name, "label": t.label, "count": len(t.positions)}
            for t in FORMATION_TEMPLATES.values()
        ]

    def get_template(self, name: str) -> list[dict] | None:
        t = FORMATION_TEMPLATES.get(name)
        if t is None:
            return None
        return [p.to_dict() if hasattr(p, "to_dict") else p for p in t.positions]

    def create_state(self, name: str = "", formation_home: str = "") -> WhiteboardState:
        state = WhiteboardState(name=name or f"Board {len(self._states) + 1}")
        if formation_home and formation_home in FORMATION_TEMPLATES:
            t = FORMATION_TEMPLATES[formation_home]
            state.players_home = [
                {"x": p["x"], "y": p["y"], "number": p["num"], "position": p["pos"], "team": "home"}
                for p in t.positions
            ]
            state.formation_home = formation_home
        self._states[state.id] = state
        return state

    def get_state(self, state_id: str) -> WhiteboardState | None:
        return self._states.get(state_id)

    def list_states(self) -> list[dict]:
        return [
            {
                "id": s.id,
                "name": s.name,
                "formation_home": s.formation_home,
                "formation_away": s.formation_away,
                "timestamp": s.timestamp,
                "annotation_count": len(s.annotations),
                "player_count": len(s.players_home) + len(s.players_away),
            }
            for s in self._states.values()
        ]

    def delete_state(self, state_id: str) -> bool:
        return self._states.pop(state_id, None) is not None

    def update_state(self, state_id: str, data: dict) -> WhiteboardState | None:
        state = self._states.get(state_id)
        if state is None:
            return None
        if "name" in data:
            state.name = str(data["name"])
        if "annotations" in data:
            state.annotations = data["annotations"]
        if "players_home" in data:
            state.players_home = data["players_home"]
        if "players_away" in data:
            state.players_away = data["players_away"]
        if "ball_position" in data:
            state.ball_position = data["ball_position"]
        if "formation_home" in data:
            state.formation_home = str(data["formation_home"])
        if "formation_away" in data:
            state.formation_away = str(data["formation_away"])
        if "pitch_orientation" in data:
            state.pitch_orientation = str(data["pitch_orientation"])
        if "metadata" in data:
            state.metadata.update(data["metadata"])
        state.timestamp = datetime.now(UTC).isoformat()
        return state

    def add_annotation(self, state_id: str, annotation: dict) -> Annotation | None:
        state = self._states.get(state_id)
        if state is None:
            return None
        a = Annotation(
            type=annotation.get("type", "arrow"),
            points=annotation.get("points", []),
            color=annotation.get("color", "#ffffff"),
            width=annotation.get("width", 3.0),
            opacity=annotation.get("opacity", 1.0),
            dashed=annotation.get("dashed", False),
            label=annotation.get("label", ""),
            player_number=annotation.get("player_number", ""),
            layer=annotation.get("layer", 0),
        )
        state.annotations.append(a.to_dict())
        return a

    def remove_annotation(self, state_id: str, annotation_id: str) -> bool:
        state = self._states.get(state_id)
        if state is None:
            return False
        before = len(state.annotations)
        state.annotations = [a for a in state.annotations if a.get("id") != annotation_id]
        return len(state.annotations) < before

    def clear_annotations(self, state_id: str) -> bool:
        state = self._states.get(state_id)
        if state is None:
            return False
        state.annotations.clear()
        return True

    def set_players_from_formation(
        self,
        state_id: str,
        formation_name: str,
        team: str = "home",
    ) -> list[dict] | None:
        state = self._states.get(state_id)
        if state is None:
            return None
        t = FORMATION_TEMPLATES.get(formation_name)
        if t is None:
            return None
        players = [
            {"x": p["x"], "y": p["y"], "number": p["num"], "position": p["pos"], "team": team}
            for p in t.positions
        ]
        if team == "home":
            state.players_home = players
            state.formation_home = formation_name
        else:
            state.players_away = players
            state.formation_away = formation_name
        return players

    def move_player(
        self, state_id: str, player_index: int, x: float, y: float, team: str = "home"
    ) -> bool:
        state = self._states.get(state_id)
        if state is None:
            return False
        roster = state.players_home if team == "home" else state.players_away
        if 0 <= player_index < len(roster):
            roster[player_index]["x"] = x
            roster[player_index]["y"] = y
            return True
        return False

    def generate_svg(self, state_id: str, width: int = 600, height: int = 400) -> str | None:
        """Generate an SVG string of the whiteboard state."""
        state = self._states.get(state_id)
        if state is None:
            return None

        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
        svg += '  <rect width="100%" height="100%" fill="#1a5c2a"/>\n'

        sx, sy = width / self.PITCH_W, height / self.PITCH_H

        svg += "  <!-- Pitch outline -->\n"
        svg += f'  <rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="white" stroke-width="2"/>\n'
        cx, cy = width / 2, height / 2
        svg += f'  <line x1="{cx}" y1="0" x2="{cx}" y2="{height}" stroke="white" stroke-width="1.5" stroke-dasharray="8,4"/>\n'
        svg += f'  <circle cx="{cx}" cy="{cy}" r="15" fill="none" stroke="white" stroke-width="1.5"/>\n'

        gk_y = height / 2
        svg += f'  <rect x="0" y="{gk_y - 20}" width="10" height="40" fill="none" stroke="white" stroke-width="1.5"/>\n'
        svg += f'  <rect x="{width - 10}" y="{gk_y - 20}" width="10" height="40" fill="none" stroke="white" stroke-width="1.5"/>\n'

        for p in state.players_home:
            px, py = p["x"] * sx, p["y"] * sy
            num = p.get("number", "")
            svg += f'  <circle cx="{px}" cy="{py}" r="8" fill="#e74c3c" stroke="white" stroke-width="1.5"/>\n'
            svg += f'  <text x="{px}" y="{py + 1}" fill="white" font-size="7" text-anchor="middle" dominant-baseline="middle">{num}</text>\n'

        for p in state.players_away:
            px, py = p["x"] * sx, p["y"] * sy
            num = p.get("number", "")
            svg += f'  <circle cx="{px}" cy="{py}" r="8" fill="#3498db" stroke="white" stroke-width="1.5"/>\n'
            svg += f'  <text x="{px}" y="{py + 1}" fill="white" font-size="7" text-anchor="middle" dominant-baseline="middle">{num}</text>\n'

        if state.ball_position:
            bx = state.ball_position.get("x", 50) * sx
            by = state.ball_position.get("y", 50) * sy
            svg += f'  <circle cx="{bx}" cy="{by}" r="5" fill="white" stroke="black" stroke-width="1"/>\n'

        annos = sorted(state.annotations, key=lambda a: a.get("layer", 0))
        for a in annos:
            svg += self._annotation_to_svg(a, sx, sy)

        svg += "</svg>"
        return svg

    def _annotation_to_svg(self, a: dict, sx: float, sy: float) -> str:
        atype = a.get("type", "line")
        pts = a.get("points", [])
        color = a.get("color", "#ffffff")
        width = a.get("width", 2.0)
        opacity = a.get("opacity", 1.0)
        dashed = 'stroke-dasharray="6,3"' if a.get("dashed") else ""
        label = a.get("label", "")

        if atype == "line" and len(pts) >= 2:
            x1, y1 = pts[0]["x"] * sx, pts[0]["y"] * sy
            x2, y2 = pts[1]["x"] * sx, pts[1]["y"] * sy
            return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" opacity="{opacity}" {dashed}/>\n'

        if atype == "arrow" and len(pts) >= 2:
            x1, y1 = pts[0]["x"] * sx, pts[0]["y"] * sy
            x2, y2 = pts[1]["x"] * sx, pts[1]["y"] * sy
            angle = math.atan2(y2 - y1, x2 - x1)
            arrow_len = 10
            ax1 = x2 - arrow_len * math.cos(angle - 0.5)
            ay1 = y2 - arrow_len * math.sin(angle - 0.5)
            ax2 = x2 - arrow_len * math.cos(angle + 0.5)
            ay2 = y2 - arrow_len * math.sin(angle + 0.5)
            result = f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" opacity="{opacity}" {dashed}/>\n'
            result += f'  <polygon points="{x2},{y2} {ax1},{ay1} {ax2},{ay2}" fill="{color}" opacity="{opacity}"/>\n'
            return result

        if atype == "circle" and len(pts) >= 1:
            cx, cy = pts[0]["x"] * sx, pts[0]["y"] * sy
            r = pts[0].get("radius", 15) * min(sx, sy)
            return f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}" {dashed}/>\n'

        if atype == "rectangle" and len(pts) >= 2:
            x1, y1 = pts[0]["x"] * sx, pts[0]["y"] * sy
            x2, y2 = pts[1]["x"] * sx, pts[1]["y"] * sy
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            return f'  <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}" {dashed}/>\n'

        if atype == "freehand" and len(pts) >= 2:
            d = " ".join(
                f"L{p['x'] * sx},{p['y'] * sy}" if i else f"M{p['x'] * sx},{p['y'] * sy}"
                for i, p in enumerate(pts)
            )
            return f'  <path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}" {dashed}/>\n'

        if atype == "text" and len(pts) >= 1:
            tx, ty = pts[0]["x"] * sx, pts[0]["y"] * sy
            return f'  <text x="{tx}" y="{ty}" fill="{color}" font-size="{width * 4}" opacity="{opacity}">{label}</text>\n'

        if atype == "zone" and len(pts) >= 4:
            d = " ".join(
                f"L{p['x'] * sx},{p['y'] * sy}" if i else f"M{p['x'] * sx},{p['y'] * sy}"
                for i, p in enumerate(pts)
            )
            d += " Z"
            return f'  <path d="{d}" fill="{color}" fill-opacity="{opacity * 0.3}" stroke="{color}" stroke-width="{width}"/>\n'

        return ""

    def generate_player_run(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        color: str = "#f39c12",
        label: str = "",
    ) -> Annotation:
        pts = [{"x": start_x, "y": start_y}, {"x": end_x, "y": end_y}]
        return Annotation(type="arrow", points=pts, color=color, width=3, label=label)

    def generate_pass(
        self, start_x: float, start_y: float, end_x: float, end_y: float, color: str = "#2ecc71"
    ) -> Annotation:
        pts = [{"x": start_x, "y": start_y}, {"x": end_x, "y": end_y}]
        return Annotation(type="arrow", points=pts, color=color, width=2, dashed=True)

    def generate_shot(
        self, x: float, y: float, target_x: float = 100, target_y: float = 50
    ) -> Annotation:
        pts = [{"x": x, "y": y}, {"x": target_x, "y": target_y}]
        return Annotation(type="arrow", points=pts, color="#e74c3c", width=3)

    def to_json(self, state_id: str) -> str | None:
        state = self.get_state(state_id)
        if state is None:
            return None
        return json.dumps(state.to_dict())
