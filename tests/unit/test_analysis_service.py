"""Tests for AnalysisService - statistics computation."""

from __future__ import annotations

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.analysis_service import (
    AnalysisService,
)
from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData


def make_detection(
    track_id: int,
    class_name: str,
    x: float,
    y: float,
    w: float = 20.0,
    h: float = 40.0,
    confidence: float = 0.9,
) -> Detection:
    """Helper to create a Detection with sensible defaults."""
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
    """Helper to create a FrameDetections."""
    return FrameDetections(
        frame_number=frame_number,
        timestamp=timestamp,
        detections=detections,
        image_width=width,
        image_height=height,
    )


@pytest.mark.asyncio
async def test_player_stats_distance() -> None:
    """Test that player distance is computed from positions."""
    service = AnalysisService()

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
    )

    analysis = await service.analyze_match(track_data, match_id=1)

    assert 1 in analysis.players
    player = analysis.players[1]
    assert player.distance_covered_m > 0
    assert player.max_speed_kmh > 0


@pytest.mark.asyncio
async def test_possession_split() -> None:
    """Possession split requires real team assignment.

    With player_teams present, the split must be attributed to the actual
    teams. Without any team assignment, possession must NOT be fabricated
    from track-ID parity (the old behavior invented a 50/50 split from
    tid % 2, which has no relation to real team membership) -- an honest
    zero beats a made-up number.
    """
    service = AnalysisService()

    frames = []
    for i in range(20):
        ball_x = 100 + (i * 5) % 200
        player_id = 1 if i % 2 == 0 else 2
        player_x = ball_x + 5
        ball = make_detection(track_id=99, class_name="sports ball", x=ball_x, y=300, w=5, h=5)
        player = make_detection(track_id=player_id, class_name="person", x=player_x, y=300)
        frames.append(make_frame(i, i * 0.1, [ball, player]))

    # With real team assignment: track 1 = home, track 2 = away. Ball sits
    # next to each of them half the frames, so both teams get a share.
    track_data = MatchTrackData(
        match_id=1,
        fps=10.0,
        total_frames=20,
        duration_seconds=2.0,
        frames=frames,
        track_registry={1: {}, 2: {}, 99: {}},
        player_teams={1: "home", 2: "away"},
    )

    analysis = await service.analyze_match(track_data, match_id=1)

    total_possession = analysis.home_team.possession_pct + analysis.away_team.possession_pct
    assert 99 <= total_possession <= 101

    # Without team assignment: no fabricated 50/50 -- honest zeros.
    track_data_no_teams = MatchTrackData(
        match_id=1,
        fps=10.0,
        total_frames=20,
        duration_seconds=2.0,
        frames=frames,
        track_registry={1: {}, 2: {}, 99: {}},
    )
    analysis_no_teams = await service.analyze_match(track_data_no_teams, match_id=1)
    assert analysis_no_teams.home_team.possession_pct == 0.0
    assert analysis_no_teams.away_team.possession_pct == 0.0


@pytest.mark.asyncio
async def test_pass_detection() -> None:
    """Test pass detection when ball possession changes."""
    service = AnalysisService()

    frames = []
    ball_x, ball_y = 100, 300

    for i in range(30):
        ball = make_detection(track_id=99, class_name="sports ball", x=ball_x, y=ball_y, w=5, h=5)

        if i < 10:
            player = make_detection(track_id=1, class_name="person", x=ball_x + 5, y=ball_y)
        elif i < 20:
            player = make_detection(track_id=2, class_name="person", x=ball_x + 5, y=ball_y)
        else:
            player = make_detection(track_id=1, class_name="person", x=ball_x + 5, y=ball_y)

        frames.append(make_frame(i, i * 0.1, [ball, player]))

    track_data = MatchTrackData(
        match_id=1,
        fps=10.0,
        total_frames=30,
        duration_seconds=3.0,
        frames=frames,
        track_registry={1: {}, 2: {}, 99: {}},
    )

    analysis = await service.analyze_match(track_data, match_id=1)

    pass_events = [e for e in analysis.events if e["type"] == "pass"]
    assert len(pass_events) > 0


@pytest.mark.asyncio
async def test_confidence_calculation() -> None:
    """Test confidence score is computed."""
    service = AnalysisService()

    frames = []
    for i in range(10):
        ball = make_detection(track_id=99, class_name="sports ball", x=100, y=300, w=5, h=5)
        player = make_detection(track_id=1, class_name="person", x=110, y=300)
        frames.append(make_frame(i, i * 0.1, [ball, player]))

    track_data = MatchTrackData(
        match_id=1,
        fps=10.0,
        total_frames=10,
        duration_seconds=1.0,
        frames=frames,
        track_registry={1: {}, 99: {}},
    )

    analysis = await service.analyze_match(track_data, match_id=1)

    assert 0 <= analysis.confidence_overall <= 1
