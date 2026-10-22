from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

STREAM_TYPES = {
    "m3u8": r"\.m3u8",
    "rtmp": r"^rtmp://",
    "youtube": r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/live)",
    "twitch": r"(twitch\.tv/)",
}


class LiveStreamCaptureService:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or tempfile.gettempdir()) / "kawkab_streams"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen] = {}
        self._recording: dict[str, bool] = {}
        self._chapter_markers: dict[str, list[dict]] = {}

    def detect_source_type(self, url: str) -> str:
        for stype, pattern in STREAM_TYPES.items():
            if re.search(pattern, url, re.IGNORECASE):
                return stype
        return "unknown"

    def start_capture(self, url: str, stream_id: str = "", output_filename: str = "") -> str:
        try:
            sid = stream_id or f"stream_{int(time.time())}"
            if sid in self._processes and self._processes[sid].poll() is None:
                return json.dumps({"error": "Stream already capturing", "stream_id": sid})

            filename = output_filename or f"{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = self.output_dir / filename

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                url,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-f",
                "mp4",
                str(output_path),
            ]

            popen_kwargs = {}
            if hasattr(subprocess, "CREATE_NO_WINDOW"):  # Windows-only flag
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                **popen_kwargs,
            )
            self._processes[sid] = proc
            self._recording[sid] = True
            self._chapter_markers[sid] = []

            logger.info(f"Stream capture started: {sid} -> {output_path}")
            return json.dumps(
                {
                    "ok": True,
                    "stream_id": sid,
                    "output": str(output_path),
                    "source_type": self.detect_source_type(url),
                }
            )
        except Exception as e:
            logger.error(f"start_capture failed: {e}")
            return json.dumps({"error": str(e)})

    def stop_capture(self, stream_id: str) -> str:
        try:
            proc = self._processes.get(stream_id)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self._recording[stream_id] = False
            markers = self._chapter_markers.get(stream_id, [])
            if stream_id in self._processes:
                del self._processes[stream_id]
            return json.dumps({"ok": True, "stream_id": stream_id, "chapters": markers})
        except Exception as e:
            logger.error(f"stop_capture failed: {e}")
            return json.dumps({"error": str(e)})

    def get_stream_status(self, stream_id: str) -> str:
        try:
            proc = self._processes.get(stream_id)
            running = proc is not None and proc.poll() is None
            return json.dumps(
                {
                    "stream_id": stream_id,
                    "running": running,
                    "recording": self._recording.get(stream_id, False),
                    "chapters": len(self._chapter_markers.get(stream_id, [])),
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_streams(self) -> str:
        try:
            result = []
            for sid in list(self._processes.keys()):
                proc = self._processes.get(sid)
                running = proc is not None and proc.poll() is None
                result.append(
                    {
                        "stream_id": sid,
                        "running": running,
                        "chapters": len(self._chapter_markers.get(sid, [])),
                    }
                )
            return json.dumps({"streams": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def add_chapter_marker(self, stream_id: str, label: str = "") -> str:
        try:
            if stream_id not in self._chapter_markers:
                self._chapter_markers[stream_id] = []
            marker = {
                "time": time.time(),
                "label": label or f"Marker {len(self._chapter_markers[stream_id]) + 1}",
            }
            self._chapter_markers[stream_id].append(marker)
            return json.dumps({"ok": True, "marker": marker})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_recordings(self) -> str:
        try:
            files = sorted(self.output_dir.glob("*.mp4"), key=os.path.getmtime, reverse=True)
            return json.dumps(
                {
                    "recordings": [
                        {
                            "path": str(f),
                            "name": f.name,
                            "size": f.stat().st_size,
                            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                        }
                        for f in files[:50]
                    ]
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})


class BroadcastOCRTagger:
    """OCR-based auto-tagging from broadcast scoreboard overlays."""

    def __init__(self):
        self._enabled = False

    def start(self):
        self._enabled = True

    def stop(self):
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def process_frame(self, frame) -> dict | None:
        """Analyze a video frame for scoreboard overlay text.
        Returns detected event dict or None.
        """
        if not self._enabled or frame is None:
            return None
        try:
            import cv2

            h, w = frame.shape[:2]
            # Scoreboard is typically in the top-left or top-center
            roi = frame[0 : int(h * 0.12), int(w * 0.05) : int(w * 0.45)]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Use pytesseract if available
            try:
                import pytesseract

                text = pytesseract.image_to_string(thresh, config="--psm 7").strip()
                if text:
                    return {"text": text, "source": "scoreboard"}
            except ImportError:
                pass

            # Simple heuristic: detect goal flash (screen-wide white flash)
            brightness = cv2.mean(frame)[0]
            if brightness > 240:
                return {"text": "possible_goal", "source": "flash_detection"}
            return None
        except Exception:
            return None
