"""Real-time analysis mode for Kawkab AI.

Processes a live video stream (file-based with low-latency mode, HTTP,
RTSP, or webcam) and emits frame-by-frame analytics with alert hooks.

Design goals:
- Frame-rate aware: process every Nth frame to maintain target FPS
- Alert system: configurable callbacks for events (e.g. shot, offside, card)
- Buffer-bounded: cap in-memory frame history to keep memory low
- Backpressure-aware: if processing falls behind, drop frames
- Cooperative cancel: external cancel() can stop a running stream

The service sits on top of CVService.process_frame and emits
``RealtimeEvent`` instances (typed dataclass) through subscribers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cv_service import CVService, FrameDetections

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class AlertSeverity(StrEnum):
    """Alert severity levels for real-time events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertKind(StrEnum):
    """Types of real-time alerts.

    Extend this enum when adding new alert sources. Custom alert kinds
    can use the 'custom.<name>' convention for application-specific events.
    """

    SHOT = "shot"
    GOAL = "goal"
    OFFSIDE = "offside"
    POSSESSION_CHANGE = "possession_change"
    TACKLE = "tackle"
    PLAYER_DOWN = "player_down"
    BALL_OUT = "ball_out"
    CROWD_SURGE = "crowd_surge"
    LOW_CONFIDENCE = "low_confidence"
    LOW_FPS = "low_fps"
    TRACK_LOST = "track_lost"
    NEW_PERIOD = "new_period"


@dataclass
class RealtimeEvent:
    """A single real-time analytics event."""

    kind: AlertKind
    severity: AlertSeverity
    timestamp_s: float
    frame_index: int
    team: str | None = None
    player_track_id: int | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamStats:
    """Live stream statistics."""

    source: str
    target_fps: float
    actual_fps: float
    frames_processed: int
    frames_dropped: int
    total_frames: int | None
    elapsed_s: float
    events_emitted: int
    avg_track_count: float
    low_confidence_frames: int


def _resolve_frame(f: Any) -> Any:
    """Helper for type-checked access to FrameDetections fields."""
    return f


class RealtimeSubscriber:
    """Subscriber interface for real-time events.

    Implementations should be cheap to invoke (no I/O blocking).
    For heavy work, dispatch to a queue and process in a worker.
    """

    async def on_event(self, event: RealtimeEvent) -> None:
        raise NotImplementedError

    async def on_stats(self, stats: StreamStats) -> None:
        pass


class AlertRule:
    """Base class for alert detection rules.

    Subclass and implement :meth:`evaluate` to detect specific events from
    incoming ``FrameDetections`` and stream context.
    """

    kind: AlertKind = AlertKind.SHOT
    cooldown_s: float = 2.0

    def __init__(self) -> None:
        self._last_fired_at: float = -1e9

    def evaluate(
        self,
        frame: FrameDetections,
        prev_frame: FrameDetections | None,
        ctx: dict[str, Any],
    ) -> RealtimeEvent | None:
        raise NotImplementedError

    def can_fire(self, now: float) -> bool:
        return (now - self._last_fired_at) >= self.cooldown_s

    def mark_fired(self, now: float) -> None:
        self._last_fired_at = now


class ShotAlertRule(AlertRule):
    """Detect when a shot event is present in the frame metadata."""

    kind = AlertKind.SHOT
    cooldown_s = 1.5

    def evaluate(
        self,
        frame: FrameDetections,
        prev_frame: FrameDetections | None,
        ctx: dict[str, Any],
    ) -> RealtimeEvent | None:
        is_shot = bool(getattr(frame, "is_shot_frame", False))
        if not is_shot:
            return None
        if prev_frame is not None and bool(getattr(prev_frame, "is_shot_frame", False)):
            return None
        now = float(getattr(frame, "timestamp", 0.0))
        if not self.can_fire(now):
            return None
        self.mark_fired(now)
        return RealtimeEvent(
            kind=self.kind,
            severity=AlertSeverity.WARNING,
            timestamp_s=now,
            frame_index=frame.frame_number,
            message="Shot detected",
            payload={"ball_pos": getattr(frame, "ball_position", None)},
        )


class LowFpsAlertRule(AlertRule):
    """Fire when the actual stream FPS drops below a threshold."""

    kind = AlertKind.LOW_FPS
    cooldown_s = 5.0

    def __init__(self, min_fps: float = 10.0) -> None:
        super().__init__()
        self.min_fps = min_fps

    def evaluate(
        self,
        frame: FrameDetections,
        prev_frame: FrameDetections | None,
        ctx: dict[str, Any],
    ) -> RealtimeEvent | None:
        actual_fps = float(ctx.get("actual_fps", 0.0))
        if actual_fps >= self.min_fps:
            return None
        now = float(getattr(frame, "timestamp", 0.0))
        if not self.can_fire(now):
            return None
        self.mark_fired(now)
        return RealtimeEvent(
            kind=self.kind,
            severity=AlertSeverity.WARNING,
            timestamp_s=now,
            frame_index=frame.frame_number,
            message=f"Stream FPS low: {actual_fps:.1f}",
            payload={"actual_fps": actual_fps, "target_fps": ctx.get("target_fps")},
        )


class LowConfidenceAlertRule(AlertRule):
    """Fire when the average detection confidence drops below a threshold."""

    kind = AlertKind.LOW_CONFIDENCE
    cooldown_s = 3.0

    def __init__(self, min_conf: float = 0.4) -> None:
        super().__init__()
        self.min_conf = min_conf

    def evaluate(
        self,
        frame: FrameDetections,
        prev_frame: FrameDetections | None,
        ctx: dict[str, Any],
    ) -> RealtimeEvent | None:
        avg_conf = float(ctx.get("avg_confidence", 1.0))
        if avg_conf >= self.min_conf:
            return None
        now = float(getattr(frame, "timestamp", 0.0))
        if not self.can_fire(now):
            return None
        self.mark_fired(now)
        return RealtimeEvent(
            kind=self.kind,
            severity=AlertSeverity.INFO,
            timestamp_s=now,
            frame_index=frame.frame_number,
            message=f"Detection confidence low: {avg_conf:.2f}",
            payload={"avg_confidence": avg_conf},
        )


class RealtimeService:
    """Process a live stream and emit analytics events.

    Args:
        cv_service: The computer-vision service to use for detection.
        target_fps: Target processing rate. Frames arriving faster are dropped.
        buffer_size: How many recent frames to keep in memory (for replays).
        stats_interval_s: How often to emit ``StreamStats`` to subscribers.
    """

    def __init__(
        self,
        cv_service: CVService | None = None,
        target_fps: float = 15.0,
        buffer_size: int = 300,
        stats_interval_s: float = 1.0,
    ) -> None:
        self.cv_service = cv_service or CVService()
        self.target_fps = float(target_fps)
        self.buffer_size = int(buffer_size)
        self.stats_interval_s = float(stats_interval_s)
        self._subscribers: list[RealtimeSubscriber] = []
        self._alert_rules: list[AlertRule] = [
            ShotAlertRule(),
            LowFpsAlertRule(),
            LowConfidenceAlertRule(),
        ]
        self._cancel = False

    def subscribe(self, subscriber: RealtimeSubscriber) -> None:
        """Register a subscriber to receive events and stats."""
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: RealtimeSubscriber) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def add_alert_rule(self, rule: AlertRule) -> None:
        self._alert_rules.append(rule)

    def cancel(self) -> None:
        """Cancel a running stream cooperatively."""
        self._cancel = True

    async def run_file(
        self,
        video_path: Path,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> StreamStats:
        """Process a video file in real-time mode.

        Frames are processed at ``target_fps`` regardless of source FPS.
        Use this for low-latency replay (e.g. coach review during a match).
        """
        import cv2

        if not self._cancel:
            await self.cv_service.initialize()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(fps / self.target_fps)) if self.target_fps > 0 and fps > 0 else 1
        return await self._run_capture(
            cap=cap,
            source=str(video_path),
            total_frames=total_frames,
            frame_skip=frame_skip,
            progress_callback=progress_callback,
        )

    async def run_webcam(
        self,
        device_index: int = 0,
    ) -> StreamStats:
        """Process a local webcam stream."""
        import cv2

        if not self._cancel:
            await self.cv_service.initialize()
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam {device_index}")
        return await self._run_capture(
            cap=cap,
            source=f"webcam:{device_index}",
            total_frames=None,
            frame_skip=1,
        )

    async def _run_capture(
        self,
        cap: Any,
        source: str,
        total_frames: int | None,
        frame_skip: int,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> StreamStats:
        """Core loop: read frames, run alert rules, dispatch events."""
        buffer: deque[FrameDetections] = deque(maxlen=self.buffer_size)
        frames_processed = 0
        frames_dropped = 0
        events_emitted = 0
        conf_sum = 0.0
        conf_count = 0
        low_conf_frames = 0
        prev_frame: FrameDetections | None = None
        last_stats_t = time.monotonic()
        start_t = last_stats_t
        last_frame_t = start_t
        try:
            while not self._cancel:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames_dropped += frame_skip - 1
                t_now = time.monotonic()
                if t_now - last_frame_t < (1.0 / self.target_fps if self.target_fps > 0 else 0.0):
                    frames_dropped += 1
                    continue
                last_frame_t = t_now
                detection = await self.cv_service.detect_frame(
                    bgr, frames_processed, time.monotonic() - start_t
                )
                if detection is None:
                    continue
                buffer.append(detection)
                frames_processed += 1
                if detection.detections:
                    confs = [d.confidence for d in detection.detections if d.confidence is not None]
                    if confs:
                        avg = sum(confs) / len(confs)
                        conf_sum += avg
                        conf_count += 1
                        if avg < 0.4:
                            low_conf_frames += 1
                elapsed = t_now - start_t
                actual_fps = frames_processed / elapsed if elapsed > 0 else 0.0
                avg_conf = (conf_sum / conf_count) if conf_count > 0 else 1.0
                ctx = {
                    "actual_fps": actual_fps,
                    "target_fps": self.target_fps,
                    "avg_confidence": avg_conf,
                }
                for rule in self._alert_rules:
                    event = rule.evaluate(detection, prev_frame, ctx)
                    if event is not None:
                        events_emitted += 1
                        await self._dispatch_event(event)
                prev_frame = detection
                if t_now - last_stats_t >= self.stats_interval_s:
                    last_stats_t = t_now
                    stats = StreamStats(
                        source=source,
                        target_fps=self.target_fps,
                        actual_fps=actual_fps,
                        frames_processed=frames_processed,
                        frames_dropped=frames_dropped,
                        total_frames=total_frames,
                        elapsed_s=elapsed,
                        events_emitted=events_emitted,
                        avg_track_count=float(
                            len(detection.detections) if detection.detections else 0
                        ),
                        low_confidence_frames=low_conf_frames,
                    )
                    await self._dispatch_stats(stats)
                    if progress_callback is not None and total_frames:
                        pct = int(min(99, (frames_processed * frame_skip / total_frames) * 100))
                        progress_callback(pct, f"processed {frames_processed} frames")
            elapsed = time.monotonic() - start_t
            actual_fps = frames_processed / elapsed if elapsed > 0 else 0.0
            return StreamStats(
                source=source,
                target_fps=self.target_fps,
                actual_fps=actual_fps,
                frames_processed=frames_processed,
                frames_dropped=frames_dropped,
                total_frames=total_frames,
                elapsed_s=elapsed,
                events_emitted=events_emitted,
                avg_track_count=(conf_sum / max(1, conf_count)),
                low_confidence_frames=low_conf_frames,
            )
        finally:
            cap.release()
            self._cancel = False

    async def _dispatch_event(self, event: RealtimeEvent) -> None:
        for sub in list(self._subscribers):
            try:
                await sub.on_event(event)
            except Exception as exc:
                logger.exception("Subscriber %s failed: %s", sub, exc)

    async def _dispatch_stats(self, stats: StreamStats) -> None:
        for sub in list(self._subscribers):
            try:
                await sub.on_stats(stats)
            except Exception as exc:
                logger.exception("Subscriber %s failed: %s", sub, exc)


class ConsoleSubscriber(RealtimeSubscriber):
    """Reference subscriber that prints events to stdout (debug)."""

    async def on_event(self, event: RealtimeEvent) -> None:
        logger.info(
            "EVT %s t=%.2fs frame=%d sev=%s msg=%s",
            event.kind.value,
            event.timestamp_s,
            event.frame_index,
            event.severity.value,
            event.message,
        )


class CallbackSubscriber(RealtimeSubscriber):
    """Subscriber that forwards events to a sync or async callable.

    Useful for plugging the real-time pipeline into existing UI
    callbacks, webhooks, or logging systems.
    """

    def __init__(self, callback: Callable[[RealtimeEvent], Any]) -> None:
        self._cb = callback

    async def on_event(self, event: RealtimeEvent) -> None:
        result = self._cb(event)
        if asyncio.iscoroutine(result):
            await result
