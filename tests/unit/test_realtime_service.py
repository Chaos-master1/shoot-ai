"""Tests for RealtimeService.

Focus on:
- Subscriber dispatch (sync/async, multi-subscriber)
- Alert rules (cooldown, prev-frame, threshold)
- Backpressure (target_fps frame dropping)
- Cooperative cancellation
- Stats accounting
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_rt = load_service_module("rt_test", "realtime_service.py")

RealtimeService = _rt.RealtimeService
RealtimeEvent = _rt.RealtimeEvent
StreamStats = _rt.StreamStats
AlertSeverity = _rt.AlertSeverity
AlertKind = _rt.AlertKind
AlertRule = _rt.AlertRule
ShotAlertRule = _rt.ShotAlertRule
LowFpsAlertRule = _rt.LowFpsAlertRule
LowConfidenceAlertRule = _rt.LowConfidenceAlertRule
CallbackSubscriber = _rt.CallbackSubscriber
ConsoleSubscriber = _rt.ConsoleSubscriber
RealtimeSubscriber = _rt.RealtimeSubscriber

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class FakeDetection:
    confidence: float
    bbox: tuple = (0, 0, 10, 10)
    class_id: int = 0
    class_name: str = "player"
    track_id: int | None = None
    team: str | None = None
    is_ball: bool = False


@dataclass
class FakeFrame:
    frame_number: int
    timestamp: float
    detections: list[FakeDetection]
    image_width: int = 1920
    image_height: int = 1080
    is_shot_frame: bool = False
    ball_position: tuple | None = None


class FakeCVService:
    """Test double for CVService.detect_frame."""

    def __init__(self, frames: list[FakeFrame] | None = None) -> None:
        self.frames = frames or []
        self._i = 0
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def detect_frame(self, bgr: Any) -> FakeFrame:
        if self._i >= len(self.frames):
            return None
        f = self.frames[self._i]
        self._i += 1
        return f


class CountingSubscriber(RealtimeSubscriber):
    def __init__(self) -> None:
        self.events: list[RealtimeEvent] = []
        self.stats: list[StreamStats] = []

    async def on_event(self, event: RealtimeEvent) -> None:
        self.events.append(event)

    async def on_stats(self, stats: StreamStats) -> None:
        self.stats.append(stats)


class TestSubscriber:
    def test_subscribe_unsubscribe(self) -> None:
        svc = RealtimeService(cv_service=FakeCVService())
        sub = CountingSubscriber()
        svc.subscribe(sub)
        assert sub in svc._subscribers
        svc.unsubscribe(sub)
        assert sub not in svc._subscribers

    def test_callback_subscriber_calls_sync(self) -> None:
        captured: list[RealtimeEvent] = []
        sub = CallbackSubscriber(lambda e: captured.append(e))
        evt = RealtimeEvent(
            kind=AlertKind.SHOT,
            severity=AlertSeverity.INFO,
            timestamp_s=1.0,
            frame_index=10,
        )
        asyncio.run(sub.on_event(evt))
        assert len(captured) == 1
        assert captured[0].kind == AlertKind.SHOT

    def test_callback_subscriber_calls_async(self) -> None:
        captured: list[RealtimeEvent] = []

        async def cb(e: RealtimeEvent) -> None:
            captured.append(e)

        sub = CallbackSubscriber(cb)
        evt = RealtimeEvent(
            kind=AlertKind.SHOT,
            severity=AlertSeverity.INFO,
            timestamp_s=1.0,
            frame_index=10,
        )
        asyncio.run(sub.on_event(evt))
        assert len(captured) == 1


class TestAlertRules:
    def test_shot_rule_fires_on_first_shot_frame(self) -> None:
        rule = ShotAlertRule()
        frame = FakeFrame(
            frame_number=100,
            timestamp=10.0,
            detections=[],
            is_shot_frame=True,
        )
        evt = rule.evaluate(frame, None, {})
        assert evt is not None
        assert evt.kind == AlertKind.SHOT
        assert evt.severity == AlertSeverity.WARNING

    def test_shot_rule_skips_consecutive_shot_frames(self) -> None:
        rule = ShotAlertRule()
        f1 = FakeFrame(frame_number=100, timestamp=10.0, detections=[], is_shot_frame=True)
        f2 = FakeFrame(frame_number=101, timestamp=10.1, detections=[], is_shot_frame=True)
        rule.evaluate(f1, None, {})
        evt2 = rule.evaluate(f2, f1, {})
        assert evt2 is None

    def test_shot_rule_cooldown(self) -> None:
        rule = ShotAlertRule()
        f1 = FakeFrame(frame_number=100, timestamp=10.0, detections=[], is_shot_frame=True)
        f3 = FakeFrame(frame_number=110, timestamp=10.05, detections=[], is_shot_frame=True)
        rule.evaluate(f1, None, {})
        evt2 = rule.evaluate(f3, f1, {})
        assert evt2 is None

    def test_shot_rule_non_shot_returns_none(self) -> None:
        rule = ShotAlertRule()
        frame = FakeFrame(frame_number=100, timestamp=10.0, detections=[], is_shot_frame=False)
        evt = rule.evaluate(frame, None, {})
        assert evt is None

    def test_low_fps_rule_fires_below_threshold(self) -> None:
        rule = LowFpsAlertRule(min_fps=10.0)
        frame = FakeFrame(frame_number=100, timestamp=10.0, detections=[])
        ctx = {"actual_fps": 5.0, "target_fps": 15.0}
        evt = rule.evaluate(frame, None, ctx)
        assert evt is not None
        assert "5.0" in evt.message or "low" in evt.message.lower()

    def test_low_fps_rule_silent_above_threshold(self) -> None:
        rule = LowFpsAlertRule(min_fps=10.0)
        frame = FakeFrame(frame_number=100, timestamp=10.0, detections=[])
        ctx = {"actual_fps": 25.0, "target_fps": 30.0}
        evt = rule.evaluate(frame, None, ctx)
        assert evt is None

    def test_low_confidence_rule_fires_below(self) -> None:
        rule = LowConfidenceAlertRule(min_conf=0.4)
        frame = FakeFrame(frame_number=100, timestamp=10.0, detections=[])
        ctx = {"avg_confidence": 0.2}
        evt = rule.evaluate(frame, None, ctx)
        assert evt is not None
        assert "0.20" in evt.message

    def test_low_confidence_rule_silent_above(self) -> None:
        rule = LowConfidenceAlertRule(min_conf=0.4)
        frame = FakeFrame(frame_number=100, timestamp=10.0, detections=[])
        ctx = {"avg_confidence": 0.8}
        evt = rule.evaluate(frame, None, ctx)
        assert evt is None

    def test_custom_rule_subclass(self) -> None:
        class MyRule(AlertRule):
            kind = AlertKind.POSSESSION_CHANGE

            def evaluate(self, frame, prev_frame, ctx):
                if prev_frame is None:
                    return None
                return RealtimeEvent(
                    kind=self.kind,
                    severity=AlertSeverity.INFO,
                    timestamp_s=frame.timestamp,
                    frame_index=frame.frame_number,
                    message="custom",
                )

        rule = MyRule()
        f1 = FakeFrame(frame_number=1, timestamp=0.0, detections=[])
        f2 = FakeFrame(frame_number=2, timestamp=1.0, detections=[])
        assert rule.evaluate(f1, None, {}) is None
        assert rule.evaluate(f2, f1, {}) is not None


class TestRealtimeService:
    def test_initialization(self) -> None:
        svc = RealtimeService(cv_service=FakeCVService(), target_fps=20.0)
        assert svc.target_fps == 20.0
        assert svc.buffer_size > 0
        assert len(svc._alert_rules) == 3

    def test_add_alert_rule(self) -> None:
        svc = RealtimeService(cv_service=FakeCVService())
        initial = len(svc._alert_rules)
        svc.add_alert_rule(LowFpsAlertRule())
        assert len(svc._alert_rules) == initial + 1

    def test_cancel_idempotent(self) -> None:
        svc = RealtimeService(cv_service=FakeCVService())
        svc.cancel()
        svc.cancel()
        assert svc._cancel is True

    @pytest.mark.asyncio
    async def test_dispatch_event_to_subscriber(self) -> None:
        cv = FakeCVService(
            frames=[
                FakeFrame(
                    frame_number=1,
                    timestamp=0.0,
                    detections=[
                        FakeDetection(confidence=0.9),
                    ],
                ),
            ]
        )
        svc = RealtimeService(cv_service=cv, target_fps=0.0, stats_interval_s=100.0)
        sub = CountingSubscriber()
        svc.subscribe(sub)
        await svc._dispatch_event(
            RealtimeEvent(
                kind=AlertKind.SHOT,
                severity=AlertSeverity.INFO,
                timestamp_s=0.0,
                frame_index=1,
            )
        )
        assert len(sub.events) == 1

    @pytest.mark.asyncio
    async def test_subscriber_exception_doesnt_break_dispatch(self) -> None:
        class Broken(RealtimeSubscriber):
            async def on_event(self, event):
                raise RuntimeError("boom")

        svc = RealtimeService(cv_service=FakeCVService())
        good = CountingSubscriber()
        svc.subscribe(Broken())
        svc.subscribe(good)
        await svc._dispatch_event(
            RealtimeEvent(
                kind=AlertKind.SHOT,
                severity=AlertSeverity.INFO,
                timestamp_s=0.0,
                frame_index=1,
            )
        )
        assert len(good.events) == 1

    def test_stream_stats_dataclass(self) -> None:
        stats = StreamStats(
            source="test.mp4",
            target_fps=15.0,
            actual_fps=14.5,
            frames_processed=100,
            frames_dropped=5,
            total_frames=200,
            elapsed_s=6.9,
            events_emitted=3,
            avg_track_count=22.0,
            low_confidence_frames=2,
        )
        assert stats.frames_processed == 100
        assert stats.events_emitted == 3
