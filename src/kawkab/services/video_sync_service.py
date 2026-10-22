from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from kawkab.core.logging import get_logger
from kawkab.core.security import SecurityValidator

logger = get_logger(__name__)


@dataclass
class VideoSource:
    path: str
    label: str
    offset_seconds: float = 0.0
    duration_seconds: float = 0.0
    is_master: bool = False


@dataclass
class SyncState:
    sources: list[VideoSource] = field(default_factory=list)
    master_index: int = 0
    master_duration: float = 0.0


class MultiAngleSyncService:
    def __init__(self):
        self._state: SyncState | None = None

    def load_videos(self, video_paths: list[dict]) -> str:
        validated = []
        for v in video_paths:
            path = SecurityValidator.validate_video_path(v.get("path", ""))
            if not path or not os.path.isfile(path):
                return json.dumps({"error": f"Video not found: {v.get('path')}"})
            label = v.get("label", os.path.basename(path))
            validated.append(VideoSource(path=str(path), label=label))
        if not validated:
            return json.dumps({"error": "No valid video paths"})
        import cv2

        durations = []
        for vs in validated:
            cap = cv2.VideoCapture(vs.path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vs.duration_seconds = frame_count / fps if fps > 0 else 0.0
            cap.release()
            durations.append(vs.duration_seconds)
        validated[0].is_master = True
        self._state = SyncState(
            sources=validated,
            master_index=0,
            master_duration=durations[0] if durations else 0.0,
        )
        return json.dumps(
            {
                "sources": [
                    {
                        "label": s.label,
                        "path": s.path,
                        "duration_s": round(s.duration_seconds, 1),
                        "is_master": s.is_master,
                    }
                    for s in validated
                ],
                "master_duration": round(self._state.master_duration, 1),
            }
        )

    def set_offset(self, source_index: int, offset_seconds: float) -> str:
        if not self._state or source_index < 0 or source_index >= len(self._state.sources):
            return json.dumps({"error": "Invalid source index"})
        if source_index == self._state.master_index:
            return json.dumps({"error": "Cannot offset master source"})
        self._state.sources[source_index].offset_seconds = offset_seconds
        return json.dumps({"ok": True, "source_index": source_index, "offset_s": offset_seconds})

    def get_sync_positions(self, master_time: float) -> str:
        if not self._state or not self._state.sources:
            return json.dumps({"error": "No sync state"})
        positions = []
        for i, s in enumerate(self._state.sources):
            slave_time = master_time - s.offset_seconds
            clamped = max(0.0, min(slave_time, s.duration_seconds - 0.04))
            positions.append(
                {
                    "index": i,
                    "label": s.label,
                    "path": s.path,
                    "time_s": round(clamped, 2),
                    "duration_s": round(s.duration_seconds, 1),
                }
            )
        return json.dumps({"master_time": round(master_time, 2), "positions": positions})

    def get_state(self) -> str:
        if not self._state:
            return json.dumps({"loaded": False})
        return json.dumps(
            {
                "loaded": True,
                "sources": [
                    {
                        "label": s.label,
                        "path": s.path,
                        "duration_s": s.duration_seconds,
                        "offset_s": s.offset_seconds,
                        "is_master": s.is_master,
                    }
                    for s in self._state.sources
                ],
                "master_index": self._state.master_index,
                "master_duration": self._state.master_duration,
            }
        )

    def clear(self) -> str:
        self._state = None
        return json.dumps({"ok": True})
