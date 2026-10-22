"""Tests for new v0.6.0 features: match type detection and Kalman integration."""

from __future__ import annotations

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.analysis_service import AnalysisService
from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData
from kawkab.services.kalman_smoother import PlayerPositionSmoother


def make_detection(
    track_id: int,
    class_name: str,
    x: float,
    y: float,
    w: float = 20.0,
    h: float = 40.0,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        bbox=(x, y, x + w, y + h),
        confidence=confidence,
        class_id=0 if class_name == "person" else 32,
        class_name=class_name,
        track_id=track_id,
    )


def make_frame(
    frame_number: int,
    timestamp: float,
    detections: list[Detection],
    width: int = 1280,
    height: int = 720,
) -> FrameDetections:
    return FrameDetections(
        frame_number=frame_number,
        timestamp=timestamp,
        detections=detections,
        image_width=width,
        image_height=height,
    )


@pytest.mark.asyncio
async def test_match_type_highlight_skips_kalman() -> None:
    """Kalman should NOT be used for highlight reels."""
    service = AnalysisService(use_kalman=True)

    frames = []
    for i in range(10):
        det = make_detection(track_id=1, class_name="person", x=100 + i * 10, y=200)
        frames.append(make_frame(i, i * 0.1, [det]))

    track_data = MatchTrackData(
        match_id=1,
        fps=10.0,
        total_frames=10,
        duration_seconds=1.0,
        frames=frames,
        track_registry={1: {"track_id": 1}},
        match_type="highlight",
    )

    analysis = await service.analyze_match(track_data, match_id=1)

    assert 1 in analysis.players
    player = analysis.players[1]
    assert player.distance_covered_m > 0
    # Highlight should use raw positions, not Kalman (center x=110 for first frame with x=100, w=20)
    assert player.positions[0][1] == 110.0


@pytest.mark.asyncio
async def test_kalman_smoother_produces_valid_positions() -> None:
    """Kalman smoother should produce valid positions and reasonable speeds."""
    smoother = PlayerPositionSmoother(process_noise_std=0.3, measurement_noise_std=0.8)

    # Simulate a player moving at 2 m/s in x direction
    for i in range(50):
        x = i * 0.2  # 2 m/s * 0.1s dt
        y = 0.0
        dt = 0.1
        if i == 0:
            smoother.update(x, y, 0.0)
        else:
            smoother.update(x, y, dt)

    sx, sy = smoother.get_position()
    vx, vy = smoother.get_velocity()
    speed = smoother.get_speed_mps()

    # Should be close to the final true position (9.8 m)
    assert 8.0 < sx < 12.0, f"Expected position near 10, got {sx}"
    # Speed should be close to 2 m/s
    assert 1.0 < speed < 4.0, f"Expected speed near 2, got {speed}"


@pytest.mark.asyncio
async def test_kalman_smoother_rejects_outliers() -> None:
    """Kalman smoother should reject large measurement jumps (teleport artifacts)."""
    smoother = PlayerPositionSmoother(process_noise_std=0.3, measurement_noise_std=0.8)

    # Normal movement
    for i in range(10):
        x = i * 0.2
        dt = 0.1
        if i == 0:
            smoother.update(x, 0.0, 0.0)
        else:
            smoother.update(x, 0.0, dt)

    # Now a teleport artifact (10m jump)
    smoother.update(12.0, 0.0, 0.1)
    sx, _ = smoother.get_position()

    # Should not fully follow the teleport
    assert sx < 11.0, f"Kalman should not fully follow teleport, got {sx}"


@pytest.mark.asyncio
async def test_team_assignment_from_track_data() -> None:
    """Player team should be assigned from track_data.player_teams."""
    service = AnalysisService()

    frames = []
    for i in range(10):
        det = make_detection(track_id=1, class_name="person", x=100, y=200)
        frames.append(make_frame(i, i * 0.1, [det]))

    track_data = MatchTrackData(
        match_id=1,
        fps=10.0,
        total_frames=10,
        duration_seconds=1.0,
        frames=frames,
        track_registry={1: {"track_id": 1}},
        player_teams={1: "home"},
    )

    analysis = await service.analyze_match(track_data, match_id=1)

    assert analysis.players[1].team == "home"


def test_match_track_data_has_match_type() -> None:
    """MatchTrackData should have match_type field."""
    track_data = MatchTrackData(
        match_id=1,
        fps=30.0,
        total_frames=100,
        duration_seconds=10.0,
        frames=[],
        track_registry={},
        match_type="full_match",
    )
    assert track_data.match_type == "full_match"


def test_match_track_data_default_match_type() -> None:
    """MatchTrackData should default to 'unknown'."""
    track_data = MatchTrackData(
        match_id=1,
        fps=30.0,
        total_frames=100,
        duration_seconds=10.0,
        frames=[],
        track_registry={},
    )
    assert track_data.match_type == "unknown"
