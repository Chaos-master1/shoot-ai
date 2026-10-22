"""Handler for video/realtime bridge methods, including multi-angle sync, trimming, and highlight reel."""

from __future__ import annotations

import json
import threading
import uuid

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.services.highlight_reel_service import HighlightReelService
from kawkab.services.video_sync_service import MultiAngleSyncService
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class VideoHandler(BridgeHandlerBase):
    """Handles video / real-time streaming operations for Bridge."""

    def __init__(self, bridge, services, rate_limiter=None):
        super().__init__(bridge, services, rate_limiter)
        self._sync_service = MultiAngleSyncService()
        reel_output = getattr(bridge, "_reel_output_dir", None)
        self._highlight_reel = HighlightReelService(output_dir=reel_output)
        self._reel_progress: dict[str, dict] = {}
        self._reel_lock = threading.Lock()

    @property
    def realtime_service(self):
        return self._services.get("realtime_service")

    def _check_rate_limit(self, category: str = "tracking") -> None:
        if self._rate_limiter is not None and not self._rate_limiter.acquire(category):
            raise RuntimeError(f"Rate limit exceeded for {category}")

    # --- Multi-Angle Sync ---

    def sync_load(self, videos_json: str) -> str:
        try:
            self._check_rate_limit()
            paths = json.loads(videos_json)
            return self._sync_service.load_videos(paths)
        except Exception as e:
            logger.error(f"sync_load failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def sync_set_offset(self, source_index: int, offset_seconds: float) -> str:
        try:
            self._check_rate_limit()
            return self._sync_service.set_offset(source_index, offset_seconds)
        except Exception as e:
            logger.error(f"sync_set_offset failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def sync_positions(self, master_time: float) -> str:
        try:
            self._check_rate_limit()
            return self._sync_service.get_sync_positions(master_time)
        except Exception as e:
            logger.error(f"sync_positions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def sync_state(self) -> str:
        try:
            self._check_rate_limit()
            return self._sync_service.get_state()
        except Exception as e:
            logger.error(f"sync_state failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def sync_clear(self) -> str:
        try:
            self._check_rate_limit()
            return self._sync_service.clear()
        except Exception as e:
            logger.error(f"sync_clear failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # --- Video Trimming ---

    def trim_video(
        self, video_path: str, start_seconds: float, end_seconds: float, output_name: str = ""
    ) -> str:
        try:
            self._check_rate_limit()
            path = SecurityValidator.validate_video_path(video_path)
            if not path:
                return json.dumps({"error": "Invalid video path"})
            import asyncio

            from kawkab.services.clip_service import ClipExtractionService

            svc = ClipExtractionService()
            name = output_name or f"trim_{int(start_seconds)}_{int(end_seconds)}.mp4"
            result = asyncio.run(
                svc.extract_clip(path, start_seconds, end_seconds, name, quality="high")
            )
            return json.dumps({"output": result, "ok": True})
        except Exception as e:
            logger.error(f"trim_video failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # --- Highlight Reel ---

    def reel_compose(self, clips_json: str, output_filename: str) -> str:
        try:
            self._check_rate_limit()
            import asyncio

            from kawkab.services.highlight_reel_service import ReelClip

            # The filename is joined onto the service output dir -- reject
            # separators/traversal before it reaches os.path.join (same
            # hardening as stream_start_capture).
            fn = str(output_filename or "")
            if not fn or "/" in fn or "\\" in fn or ".." in fn or not fn.endswith(".mp4"):
                return json.dumps({"error": "invalid output filename"})
            clips_data = json.loads(clips_json)
            clips = [
                ReelClip(
                    video_path=c["video_path"],
                    start_seconds=c["start_s"],
                    end_seconds=c["end_s"],
                    label=c.get("label", ""),
                )
                for c in clips_data
            ]
            result = asyncio.run(self._highlight_reel.compose_reel(clips, output_filename))
            return result
        except Exception as e:
            logger.error(f"reel_compose failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def reel_from_events(self, match_id: int, events_json: str, video_path: str) -> str:
        try:
            self._check_rate_limit()
            events = json.loads(events_json)
            reel_id = str(uuid.uuid4())[:8]
            with self._reel_lock:
                self._reel_progress[reel_id] = {
                    "status": "starting",
                    "progress": 0.0,
                    "output_path": "",
                }

            # Track reel progress
            with self._reel_lock:
                self._reel_progress[reel_id] = {
                    "status": "processing",
                    "progress": 0.3,
                    "output_path": "",
                }

            result_str = self._highlight_reel.make_reel_from_events(match_id, events, video_path)
            result = json.loads(result_str)

            with self._reel_lock:
                if "error" in result:
                    self._reel_progress[reel_id] = {
                        "status": "error",
                        "progress": 0.0,
                        "output_path": "",
                    }
                else:
                    self._reel_progress[reel_id] = {
                        "status": "complete",
                        "progress": 1.0,
                        "output_path": result.get("output_path", ""),
                    }

            result["reel_id"] = reel_id
            return json.dumps(result)
        except Exception as e:
            logger.error(f"reel_from_events failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def reel_status(self, reel_id: str) -> str:
        """Get the current progress of a highlight reel generation."""
        try:
            with self._reel_lock:
                status = self._reel_progress.get(
                    reel_id, {"status": "unknown", "progress": 0.0, "output_path": ""}
                )
            return json.dumps(status)
        except Exception as e:
            return json.dumps(
                {"status": "error", "progress": 0.0, "output_path": "", "error": str(e)}
            )

    # --- Realtime ---

    def realtime_status(self):
        self._check_rate_limit()
        if self.realtime_service is None:
            return json.dumps({"available": False, "error": "RealtimeService not initialized"})
        try:
            return json.dumps(
                {
                    "available": True,
                    "target_fps": self.realtime_service.target_fps,
                    "buffer_size": self.realtime_service.buffer_size,
                    "alert_rule_count": len(self.realtime_service._alert_rules),
                    "subscriber_count": len(self.realtime_service._subscribers),
                }
            )
        except Exception as e:
            logger.error(f"realtime_status failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def realtime_cancel(self):
        self._check_rate_limit()
        if self.realtime_service is None:
            return json.dumps({"error": "RealtimeService not initialized"})
        try:
            self.realtime_service.cancel()
            return json.dumps({"ok": True, "message": "Realtime stream cancelled"})
        except Exception as e:
            logger.error(f"realtime_cancel failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def realtime_subscribe_console(self):
        self._check_rate_limit()
        if self.realtime_service is None:
            return json.dumps({"error": "RealtimeService not initialized"})
        try:
            from kawkab.services.realtime_service import ConsoleSubscriber

            sub = ConsoleSubscriber()
            self.realtime_service.subscribe(sub)
            return json.dumps({"ok": True, "message": "Console subscriber added"})
        except Exception as e:
            logger.error(f"realtime_subscribe_console failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
