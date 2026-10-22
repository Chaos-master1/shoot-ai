"""Video review workflow: frame-accurate clips, drawing tools, tagging.

Backend for the in-app video review screen. Stores:

- Annotations (arrows, circles, free text) tied to a frame
- Tags (e.g. "build-up", "transition", "set piece") per clip
- Clips as in/out frame ranges with optional thumbnail info

Storage is in-memory; persistence is the caller's responsibility
(JSON or DB row). Designed to be small and dependency-free so it
can run inside the existing bridge without extra I/O layers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class AnnotationKind(StrEnum):
    """Type of in-frame annotation."""

    ARROW = "arrow"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    FREEHAND = "freehand"
    TEXT = "text"


class ClipTag(StrEnum):
    """Predefined clip tags (free-form strings also allowed)."""

    BUILD_UP = "build_up"
    TRANSITION = "transition"
    SET_PIECE = "set_piece"
    DEFENSIVE_BREAKDOWN = "defensive_breakdown"
    GOAL = "goal"
    CHANCE_CREATED = "chance_created"
    INDIVIDUAL_ERROR = "individual_error"
    POSITIVE_MOMENT = "positive_moment"


@dataclass
class Annotation:
    """A drawing or text annotation on a single frame."""

    annotation_id: str
    kind: AnnotationKind
    frame_number: int
    timestamp_s: float
    geometry: dict[str, Any]
    color: str = "#FFD700"
    stroke_width: int = 3
    text: str = ""
    author: str = ""


@dataclass
class Clip:
    """A clipped segment of the video with metadata."""

    clip_id: str
    title: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    tags: list[str] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    notes: str = ""
    author: str = ""
    created_at: str = ""


@dataclass
class ReviewSession:
    """All review state for a single match video."""

    session_id: str
    match_id: int
    total_frames: int
    fps: float
    clips: list[Clip] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    notes: str = ""


class VideoReviewService:
    """In-memory storage for video review state.

    Args:
        default_fps: FPS used when not provided on the review session.
    """

    def __init__(self, default_fps: float = 30.0) -> None:
        self.default_fps = default_fps
        self._sessions: dict[str, ReviewSession] = {}
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def create_session(
        self, match_id: int, total_frames: int, fps: float | None = None
    ) -> ReviewSession:
        session = ReviewSession(
            session_id=str(uuid.uuid4()),
            match_id=match_id,
            total_frames=total_frames,
            fps=fps or self.default_fps,
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ReviewSession | None:
        return self._sessions.get(session_id)

    def add_clip(
        self,
        session_id: str,
        title: str,
        start_frame: int,
        end_frame: int,
        tags: list[str] | None = None,
        notes: str = "",
        author: str = "",
    ) -> Clip | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        clip = Clip(
            clip_id=str(uuid.uuid4()),
            title=title,
            start_frame=start_frame,
            end_frame=end_frame,
            start_ts=start_frame / session.fps,
            end_ts=end_frame / session.fps,
            tags=list(tags or []),
            notes=notes,
            author=author,
        )
        session.clips.append(clip)
        return clip

    def add_annotation(
        self,
        session_id: str,
        kind: AnnotationKind,
        frame_number: int,
        geometry: dict[str, Any],
        color: str = "#FFD700",
        text: str = "",
        author: str = "",
        clip_id: str | None = None,
    ) -> Annotation | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        ann = Annotation(
            annotation_id=str(uuid.uuid4()),
            kind=kind,
            frame_number=frame_number,
            timestamp_s=frame_number / session.fps,
            geometry=dict(geometry),
            color=color,
            text=text,
            author=author,
        )
        session.annotations.append(ann)
        if clip_id is not None:
            for c in session.clips:
                if c.clip_id == clip_id:
                    c.annotations.append(ann)
                    break
        return ann

    def remove_clip(self, session_id: str, clip_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        before = len(session.clips)
        session.clips = [c for c in session.clips if c.clip_id != clip_id]
        return len(session.clips) < before

    def remove_annotation(self, session_id: str, annotation_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        before = len(session.annotations)
        session.annotations = [a for a in session.annotations if a.annotation_id != annotation_id]
        for c in session.clips:
            c.annotations = [a for a in c.annotations if a.annotation_id != annotation_id]
        return len(session.annotations) < before

    def find_clips_by_tag(self, session_id: str, tag: str) -> list[Clip]:
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return [c for c in session.clips if tag in c.tags]

    def export_session(self, session_id: str) -> dict[str, Any] | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "match_id": session.match_id,
            "total_frames": session.total_frames,
            "fps": session.fps,
            "notes": session.notes,
            "clips": [
                {
                    "clip_id": c.clip_id,
                    "title": c.title,
                    "start_frame": c.start_frame,
                    "end_frame": c.end_frame,
                    "start_ts": c.start_ts,
                    "end_ts": c.end_ts,
                    "tags": c.tags,
                    "notes": c.notes,
                    "author": c.author,
                    "annotations": [a.annotation_id for a in c.annotations],
                }
                for c in session.clips
            ],
            "annotations": [
                {
                    "annotation_id": a.annotation_id,
                    "kind": a.kind.value,
                    "frame_number": a.frame_number,
                    "timestamp_s": a.timestamp_s,
                    "geometry": a.geometry,
                    "color": a.color,
                    "text": a.text,
                    "author": a.author,
                }
                for a in session.annotations
            ],
        }

    def import_session(self, payload: dict[str, Any]) -> ReviewSession | None:
        try:
            session = ReviewSession(
                session_id=payload["session_id"],
                match_id=int(payload["match_id"]),
                total_frames=int(payload["total_frames"]),
                fps=float(payload["fps"]),
                notes=payload.get("notes", ""),
            )
            for c in payload.get("clips", []):
                session.clips.append(
                    Clip(
                        clip_id=c["clip_id"],
                        title=c["title"],
                        start_frame=int(c["start_frame"]),
                        end_frame=int(c["end_frame"]),
                        start_ts=float(c["start_ts"]),
                        end_ts=float(c["end_ts"]),
                        tags=list(c.get("tags", [])),
                        notes=c.get("notes", ""),
                        author=c.get("author", ""),
                    )
                )
            for a in payload.get("annotations", []):
                session.annotations.append(
                    Annotation(
                        annotation_id=a["annotation_id"],
                        kind=AnnotationKind(a["kind"]),
                        frame_number=int(a["frame_number"]),
                        timestamp_s=float(a["timestamp_s"]),
                        geometry=dict(a.get("geometry", {})),
                        color=a.get("color", "#FFD700"),
                        text=a.get("text", ""),
                        author=a.get("author", ""),
                    )
                )
            self._sessions[session.session_id] = session
            return session
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to import review session: %s", e)
            return None
