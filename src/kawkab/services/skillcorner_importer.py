from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SkillCornerTrackingFrame:
    frame_id: int
    timestamp: float
    players: list[dict] = field(default_factory=list)
    ball: dict | None = None


class SkillCornerImporter:
    def import_tracking_data(self, path: str | Path) -> list[SkillCornerTrackingFrame]:
        try:
            with open(str(path), encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to read SkillCorner file {path}: {exc}")
            return []
        if isinstance(data, list):
            return self._parse_frame_list(data)
        if isinstance(data, dict):
            frames = data.get("frames", data.get("tracking", []))
            if isinstance(frames, list):
                return self._parse_frame_list(frames)
            logger.warning(f"Unexpected SkillCorner format in {path}")
            return []
        return []

    def import_match_data(self, match_id: str) -> list[SkillCornerTrackingFrame]:
        raise NotImplementedError("SkillCorner API — requires subscription")

    def to_kawkab_events(self, tracking_data: list[SkillCornerTrackingFrame]) -> list[dict]:
        events: list[dict] = []
        ball_positions: list[dict] = []
        for frame in tracking_data:
            if frame.ball and isinstance(frame.ball, dict):
                bx = frame.ball.get("x", 0)
                by = frame.ball.get("y", 0)
                ball_positions.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp": frame.timestamp,
                        "x": bx,
                        "y": by,
                        "z": frame.ball.get("z", 0),
                    }
                )
        for i, bp in enumerate(ball_positions):
            if i == 0:
                continue
            prev = ball_positions[i - 1]
            dx = bp["x"] - prev["x"]
            dy = bp["y"] - prev["y"]
            dist = (dx * dx + dy * dy) ** 0.5
            near_players = self._find_nearby_players(
                tracking_data, bp["frame_id"], bp["x"], bp["y"], radius=2.0
            )
            if dist > 3.0 and near_players:
                events.append(
                    {
                        "type": "pass",
                        "timestamp": bp["timestamp"],
                        "start_x": prev["x"],
                        "start_y": prev["y"],
                        "end_x": bp["x"],
                        "end_y": bp["y"],
                        "player": near_players[0].get("track_id", ""),
                    }
                )
            elif dist > 1.0 and not near_players:
                events.append(
                    {
                        "type": "shot",
                        "timestamp": bp["timestamp"],
                        "start_x": prev["x"],
                        "start_y": prev["y"],
                        "end_x": bp["x"],
                        "end_y": bp["y"],
                    }
                )
        return events

    def _parse_frame_list(self, frames: list[dict]) -> list[SkillCornerTrackingFrame]:
        result: list[SkillCornerTrackingFrame] = []
        for i, item in enumerate(frames):
            if not isinstance(item, dict):
                continue
            try:
                frame = SkillCornerTrackingFrame(
                    frame_id=int(item.get("frame_id", item.get("id", i))),
                    timestamp=float(item.get("timestamp", item.get("time", 0.0))),
                    players=self._parse_players(item.get("players", [])),
                    ball=self._parse_ball(item.get("ball")),
                )
                result.append(frame)
            except Exception as exc:
                logger.warning(f"Skipping SkillCorner frame {i}: {exc}")
        return result

    @staticmethod
    def _parse_players(players_raw: Any) -> list[dict]:
        if not isinstance(players_raw, list):
            return []
        players: list[dict] = []
        for p in players_raw:
            if not isinstance(p, dict):
                continue
            players.append(
                {
                    "track_id": str(p.get("track_id", p.get("id", ""))),
                    "x": float(p.get("x", 0)),
                    "y": float(p.get("y", 0)),
                    "speed": float(p.get("speed", 0)),
                }
            )
        return players

    @staticmethod
    def _parse_ball(ball_raw: Any) -> dict | None:
        if not isinstance(ball_raw, dict):
            return None
        try:
            return {
                "x": float(ball_raw.get("x", 0)),
                "y": float(ball_raw.get("y", 0)),
                "z": float(ball_raw.get("z", 0)) if ball_raw.get("z") is not None else None,
            }
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _find_nearby_players(
        frames: list[SkillCornerTrackingFrame], frame_id: int, x: float, y: float, radius: float
    ) -> list[dict]:
        for frame in frames:
            if frame.frame_id == frame_id:
                nearby = []
                for p in frame.players:
                    dx = p.get("x", 0) - x
                    dy = p.get("y", 0) - y
                    if (dx * dx + dy * dy) ** 0.5 <= radius:
                        nearby.append(p)
                return nearby
        return []
