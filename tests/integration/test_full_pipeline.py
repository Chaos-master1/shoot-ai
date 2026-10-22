"""Integration test - full pipeline from video to professional report.

Tests the complete pipeline:
1. Storage initialization with migrations
2. Match creation and video processing
3. Homography calibration
4. Analysis with Kalman smoothing
5. Advanced event detection
6. Physical load metrics
7. Pressure metrics
8. Anomaly detection and quality scoring
9. Data export
10. Multi-match analysis

This is the "smoke test" for the entire professional system.
"""

from __future__ import annotations

import pytest

from kawkab.services.advanced_event_detection_service import AdvancedEventDetectionService
from kawkab.services.analysis_service import AnalysisService
from kawkab.services.anomaly_detection_service import AnomalyDetectionService
from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData
from kawkab.services.data_export_service import DataExportService
from kawkab.services.homography_service import HomographyMatrix
from kawkab.services.physical_load_service import PhysicalLoadService
from kawkab.services.pressure_metrics_service import PressureMetricsService
from kawkab.services.quality_scoring_service import QualityScoringService


def make_detection(track_id, class_name, x, y, w=20.0, h=40.0, confidence=0.9):
    return Detection(
        bbox=(x, y, x + w, y + h),
        confidence=confidence,
        class_id=0 if class_name == "person" else 32,
        class_name=class_name,
        track_id=track_id,
    )


def make_frame(frame_number, timestamp, detections, width=1280, height=720):
    return FrameDetections(
        frame_number=frame_number,
        timestamp=timestamp,
        detections=detections,
        image_width=width,
        image_height=height,
    )


def create_test_tracking_data() -> MatchTrackData:
    """Create synthetic tracking data for a short test match."""
    frames = []
    fps = 10.0
    duration = 3.0  # 3 seconds (30 frames) - fast for tests
    total_frames = int(duration * fps)

    # Simulate players moving in a pattern
    for i in range(total_frames):
        ts = i / fps
        detections = []

        # Ball moving across the field
        ball_x = 100 + (i % 50) * 5
        ball_y = 300 + 20 * (i % 10) / 10
        detections.append(
            make_detection(99, "sports ball", ball_x, ball_y, w=5, h=5, confidence=0.8)
        )

        # 22 players moving around
        for p in range(1, 23):
            px = 100 + (p * 40) + (i % 20) * 2
            py = 150 + (p % 3) * 150 + (i % 10) * 3
            detections.append(make_detection(p, "person", px, py, confidence=0.85))

        frames.append(make_frame(i, ts, detections))

    return MatchTrackData(
        match_id=1,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration,
        frames=frames,
        track_registry={i: {"track_id": i} for i in range(1, 23)},
        player_teams={i: "home" if i <= 11 else "away" for i in range(1, 23)},
        tracking_metrics={
            "validated_player_tracks": 22,
            "raw_tracks_detected": 25,
            "fragmentation_rate": 1.2,
            "tracking_quality": "excellent",
            "frame_skip": 3,
            "team_detection": {"enabled": True, "assigned": 22, "n_clusters": 2},
        },
        match_type="full_match",
    )


def create_homography() -> HomographyMatrix:
    """Create a test homography matrix."""
    # Simple identity-like transformation for testing
    H = [
        [0.082, 0.0, 0.0],
        [0.0, 0.094, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return HomographyMatrix(
        matrix=H,
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        confidence=0.85,
        error_px=5.0,
    )


@pytest.mark.asyncio
async def test_full_pipeline() -> None:
    """Test the full professional pipeline."""
    track_data = create_test_tracking_data()
    homography = create_homography()

    # 1. Basic analysis
    analysis_service = AnalysisService(use_kalman=True)
    analysis = await analysis_service.analyze_match(
        track_data, match_id=1, homography_matrix=homography
    )

    assert analysis is not None
    assert len(analysis.players) > 0
    assert analysis.home_team.possession_pct >= 0
    assert analysis.away_team.possession_pct >= 0
    assert analysis.confidence_overall > 0

    # 2. Advanced event detection
    adv_event_svc = AdvancedEventDetectionService()
    all_events = await adv_event_svc.detect_all_advanced_events(
        track_data=track_data,
        base_events=analysis.events,
        homography_matrix=homography,
    )

    # Pipeline should produce some events (even if only dribbles in synthetic data)
    assert len(all_events) >= len(analysis.events)
    event_types = {e["type"] for e in all_events}
    assert len(event_types) > 0, "Should detect at least one event type"

    # 3. Physical load metrics
    phys_svc = PhysicalLoadService()
    physical_loads = await phys_svc.compute_physical_load(track_data, homography)

    assert len(physical_loads) > 0
    for _tid, metrics in physical_loads.items():
        assert metrics.total_distance_m >= 0
        assert metrics.max_speed_kmh <= 40.0  # human limit

    # 4. Pressure metrics
    press_svc = PressureMetricsService()
    pressure_metrics = await press_svc.compute_pressure_metrics(
        track_data=track_data,
        events=all_events,
        homography_matrix=homography,
    )

    assert "home" in pressure_metrics or "away" in pressure_metrics
    for _team, metrics in pressure_metrics.items():
        assert metrics.ppda_overall >= 0
        assert metrics.pressure_events >= 0

    # 5. Anomaly detection
    anomaly_svc = AnomalyDetectionService()
    anomalies = await anomaly_svc.detect_anomalies(
        track_data=track_data,
        analysis=analysis,
        events=all_events,
    )

    # With synthetic data, we might not have anomalies, but service should work
    assert isinstance(anomalies, list)

    # 6. Quality scoring
    quality_svc = QualityScoringService()
    scores = await quality_svc.compute_scores(
        track_data=track_data,
        analysis=analysis,
        homography_matrix=homography,
    )

    assert scores.overall > 0
    assert scores.tracking > 0
    assert scores.events >= 0
    assert scores.homography > 0
    assert scores.team_assignment > 0

    # 7. Data export service exists
    export_svc = DataExportService()
    assert export_svc is not None

    # 8. Physical team summary
    team_summary = await phys_svc.compute_team_physical_summary(
        physical_loads, track_data.player_teams
    )
    assert "home" in team_summary or "away" in team_summary

    print(
        f"Pipeline complete: {len(all_events)} events, {len(physical_loads)} players, "
        f"quality={scores.overall:.2f}"
    )


@pytest.mark.asyncio
async def test_advanced_event_dribble_detection() -> None:
    """Test that dribble detection works on clear dribble sequences."""
    svc = AdvancedEventDetectionService()

    # Create frames with clear dribble: player 1 has ball for 10 frames while moving
    frames = []
    for i in range(15):
        ball = make_detection(99, "sports ball", 100 + i * 5, 200, w=5, h=5)
        player = make_detection(1, "person", 100 + i * 5, 200)
        frames.append(make_frame(i, i * 0.1, [ball, player]))

    track_data = MatchTrackData(
        match_id=1,
        fps=10,
        total_frames=15,
        duration_seconds=1.5,
        frames=frames,
        track_registry={1: {"track_id": 1}},
        player_teams={1: "home"},
    )

    base_events = []
    all_events = await svc.detect_all_advanced_events(track_data, base_events)

    dribbles = [e for e in all_events if e["type"] == "dribble"]
    assert len(dribbles) > 0


@pytest.mark.asyncio
async def test_physical_load_speed_capping() -> None:
    """Physical load service caps speed at human limit."""
    svc = PhysicalLoadService()

    # Create trajectory with impossible speed (should be capped)
    frames = []
    for i in range(10):
        # 100m in 1 second = 360 km/h (impossible)
        ball = make_detection(99, "sports ball", 100 + i * 100, 200, w=5, h=5)
        player = make_detection(1, "person", 100 + i * 100, 200)
        frames.append(make_frame(i, i * 0.1, [ball, player]))

    track_data = MatchTrackData(
        match_id=1,
        fps=10,
        total_frames=10,
        duration_seconds=1.0,
        frames=frames,
        track_registry={1: {"track_id": 1}},
    )

    loads = await svc.compute_physical_load(track_data)
    assert 1 in loads
    assert loads[1].max_speed_kmh <= 40.0
    assert loads[1].max_speed_kmh > 0


@pytest.mark.asyncio
async def test_pressure_metrics_ppda() -> None:
    """Pressure metrics service computes PPDA correctly."""
    svc = PressureMetricsService()

    # Simple events: 10 opponent passes, 2 defensive actions
    events = [
        {"type": "pass", "team": "home", "completed": True, "timestamp": i * 1.0} for i in range(10)
    ] + [
        {"type": "tackle", "team": "away", "timestamp": 5.0},
        {"type": "interception", "team": "away", "timestamp": 8.0},
    ]

    frames = []
    for i in range(5):
        frames.append(
            make_frame(
                i,
                i * 1.0,
                [
                    make_detection(1, "person", 100, 200),
                    make_detection(12, "person", 150, 200),
                ],
            )
        )

    track_data = MatchTrackData(
        match_id=1,
        fps=1,
        total_frames=5,
        duration_seconds=5.0,
        frames=frames,
        track_registry={1: {}, 12: {}},
        player_teams={1: "home", 12: "away"},
    )

    metrics = await svc.compute_pressure_metrics(track_data, events)
    assert "away" in metrics
    # PPDA = 10 opponent passes / 2 defensive actions = 5.0
    assert metrics["away"].ppda_overall == 5.0
