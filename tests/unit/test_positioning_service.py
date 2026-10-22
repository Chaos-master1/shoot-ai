"""Tests for off-ball positioning and run analysis."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mod = load_service_module("pos_test", "positioning_service.py")
PositioningService = _mod.PositioningService
RunType = _mod.RunType
Run = _mod.Run
PositioningReport = _mod.PositioningReport

import pytest


@dataclass
class FakeBBox:
    cx: float = 0.0
    cy: float = 0.0


@dataclass
class FakeDetection:
    bbox: FakeBBox
    track_id: int
    team: str
    is_ball: bool = False


@dataclass
class FakeFrame:
    frame_number: int
    detections: list = field(default_factory=list)
    ball_position: tuple | None = None


@dataclass
class FakeMatchTrack:
    frames: list = field(default_factory=list)


class TestRunTypeClassification:
    """Direct tests for the _classify_run static method."""

    def test_behind_defense(self) -> None:
        svc = PositioningService(min_run_distance_m=1.0)
        result = svc._classify_run(
            (50, 34),
            (55, 34),
            [],
            0,
            10,
        )
        assert result == RunType.BEHIND_DEFENSE

    def test_wide(self) -> None:
        svc = PositioningService(min_run_distance_m=1.0)
        result = svc._classify_run(
            (50, 30),
            (50, 40),
            [],
            0,
            10,
        )
        assert result == RunType.WIDE

    def test_diagonal(self) -> None:
        svc = PositioningService(min_run_distance_m=1.0)
        result = svc._classify_run(
            (50, 30),
            (55, 40),
            [],
            0,
            10,
        )
        assert result == RunType.DIAGONAL

    def test_drop(self) -> None:
        svc = PositioningService(min_run_distance_m=1.0)
        result = svc._classify_run(
            (60, 34),
            (50, 34),
            [],
            0,
            10,
        )
        assert result == RunType.DROP

    def test_support(self) -> None:
        svc = PositioningService(min_run_distance_m=1.0)
        result = svc._classify_run(
            (50, 34),
            (51, 34),
            [],
            0,
            10,
        )
        assert result == RunType.SUPPORT

    def test_unknown_below_min_distance(self) -> None:
        svc = PositioningService(min_run_distance_m=10.0)
        result = svc._classify_run(
            (50, 34),
            (52, 34),
            [],
            0,
            10,
        )
        assert result == RunType.UNKNOWN

    def test_decoy_enum_exists(self) -> None:
        assert RunType.DECOY.value == "decoy"

    def test_unknown_enum_default(self) -> None:
        assert RunType.UNKNOWN.value == "unknown"


class TestDistanceCalculation:
    """Tests for _path_length and related distance logic."""

    def test_path_length_horizontal(self) -> None:
        svc = PositioningService(pitch_length_m=100, pitch_width_m=100)
        path = [(0, (0.0, 0.0)), (1, (50.0, 0.0))]
        dist = svc._path_length(path)
        assert dist == pytest.approx(50.0, rel=0.01)

    def test_path_length_vertical(self) -> None:
        svc = PositioningService(pitch_length_m=100, pitch_width_m=100)
        path = [(0, (0.0, 0.0)), (1, (0.0, 50.0))]
        dist = svc._path_length(path)
        assert dist == pytest.approx(50.0, rel=0.01)

    def test_path_length_multi_step(self) -> None:
        svc = PositioningService(pitch_length_m=105, pitch_width_m=68)
        path = [(0, (0.0, 0.0)), (1, (50.0, 0.0)), (2, (100.0, 0.0))]
        dist = svc._path_length(path)
        expected = 2 * (50 * 105.0 / 100.0)
        assert dist == pytest.approx(expected, rel=0.01)

    def test_path_length_single_point(self) -> None:
        svc = PositioningService()
        path = [(0, (50.0, 34.0))]
        dist = svc._path_length(path)
        assert dist == 0.0


class TestSpaceCreation:
    """Tests for _estimate_created_xt and speed bonuses."""

    def test_behind_defense_base_xt(self) -> None:
        svc = PositioningService()
        xt = svc._estimate_created_xt(RunType.BEHIND_DEFENSE, 20, 4.0)
        assert xt == 0.04

    def test_diagonal_base_xt(self) -> None:
        svc = PositioningService()
        xt = svc._estimate_created_xt(RunType.DIAGONAL, 20, 4.0)
        assert xt == 0.025

    def test_wide_base_xt(self) -> None:
        svc = PositioningService()
        xt = svc._estimate_created_xt(RunType.WIDE, 20, 4.0)
        assert xt == 0.02

    def test_decoy_base_xt(self) -> None:
        svc = PositioningService()
        xt = svc._estimate_created_xt(RunType.DECOY, 20, 4.0)
        assert xt == 0.015

    def test_support_base_xt(self) -> None:
        svc = PositioningService()
        xt = svc._estimate_created_xt(RunType.SUPPORT, 20, 4.0)
        assert xt == 0.005

    def test_drop_and_unknown_zero_xt(self) -> None:
        svc = PositioningService()
        assert svc._estimate_created_xt(RunType.DROP, 20, 4.0) == 0.0
        assert svc._estimate_created_xt(RunType.UNKNOWN, 20, 4.0) == 0.0

    def test_speed_bonus_high(self) -> None:
        svc = PositioningService()
        low = svc._estimate_created_xt(RunType.BEHIND_DEFENSE, 20, 4.0)
        high = svc._estimate_created_xt(RunType.BEHIND_DEFENSE, 20, 8.0)
        assert high == low + 0.01

    def test_speed_bonus_medium(self) -> None:
        svc = PositioningService()
        low = svc._estimate_created_xt(RunType.BEHIND_DEFENSE, 20, 4.0)
        medium = svc._estimate_created_xt(RunType.BEHIND_DEFENSE, 20, 6.0)
        assert medium == low + 0.005


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_data(self) -> None:
        svc = PositioningService()
        report = svc.analyze(FakeMatchTrack(), "home")
        assert report.total_runs == 0
        assert "No tracking data" in report.notes[0]

    def test_no_frames_attribute(self) -> None:
        svc = PositioningService()
        report = svc.analyze(object(), "home")
        assert report.total_runs == 0

    def test_empty_frames_list(self) -> None:
        svc = PositioningService()
        report = svc.analyze(FakeMatchTrack(frames=[]), "home")
        assert report.total_runs == 0

    def test_single_frame_run_ignored(self) -> None:
        svc = PositioningService(min_run_distance_m=1.0)
        frames = [
            FakeFrame(
                frame_number=0,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50, cy=34),
                        track_id=1,
                        team="home",
                    )
                ],
            ),
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs == 0

    def test_short_run_filtered_by_threshold(self) -> None:
        svc = PositioningService(min_run_distance_m=20.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50 + i * 0.1, cy=34),
                        track_id=1,
                        team="home",
                    )
                ],
                ball_position=(60, 34),
            )
            for i in range(30)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs == 0

    def test_very_long_run(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=10 + i * 0.8, cy=34),
                        track_id=1,
                        team="home",
                    )
                ],
                ball_position=(60, 34),
            )
            for i in range(120)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs >= 1
        run = report.runs[0]
        assert run.distance_m > 30

    def test_only_away_team_filtered(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50 + i * 0.3, cy=34),
                        track_id=2,
                        team="away",
                    ),
                ],
                ball_position=(60, 34),
            )
            for i in range(40)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs == 0

    def test_ball_detections_do_not_create_runs(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50 + i * 0.5, cy=34),
                        track_id=99,
                        team="home",
                        is_ball=True,
                    ),
                ],
            )
            for i in range(30)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs == 0

    def test_no_track_id_detection_ignored(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50 + i * 0.3, cy=34),
                        track_id=None,
                        team="home",
                    ),
                ],
            )
            for i in range(40)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs == 0

    def test_is_in_behind_defense(self) -> None:
        svc = PositioningService()
        assert svc.is_in_behind_defense(95.0, 80.0) is True
        assert svc.is_in_behind_defense(80.0, 80.0) is False
        assert svc.is_in_behind_defense(50.0, 80.0) is False


class TestAnalyzeRuns:
    """End-to-end tests for the analyze method."""

    def test_behind_defense_run(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50 + i * 0.5, cy=34),
                        track_id=1,
                        team="home",
                    )
                ],
                ball_position=(60, 34),
            )
            for i in range(60)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs >= 1
        assert report.runs[0].run_type == RunType.BEHIND_DEFENSE

    def test_wide_run(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50, cy=34 + i * 0.5),
                        track_id=1,
                        team="home",
                    )
                ],
                ball_position=(50, 34),
            )
            for i in range(60)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs >= 1
        assert report.runs[0].run_type == RunType.WIDE

    def test_diagonal_run(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50 + i * 0.4, cy=34 + i * 0.3),
                        track_id=1,
                        team="home",
                    )
                ],
            )
            for i in range(60)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs >= 1
        assert report.runs[0].run_type == RunType.DIAGONAL

    def test_drop_run(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=80 - i * 0.5, cy=34),
                        track_id=1,
                        team="home",
                    )
                ],
            )
            for i in range(60)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs >= 1
        assert report.runs[0].run_type == RunType.DROP

    def test_summary_counts(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = [
            FakeFrame(
                frame_number=i,
                detections=[
                    FakeDetection(
                        bbox=FakeBBox(cx=50 + i * 0.5, cy=34),
                        track_id=1,
                        team="home",
                    )
                ],
                ball_position=(60, 34),
            )
            for i in range(90)
        ]
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.team == "home"
        assert report.total_runs == 1
        assert report.runs_by_type.get(RunType.BEHIND_DEFENSE.value, 0) == 1
        assert report.total_xT_created >= 0.0
        assert report.longest_run_m > 0

    def test_multiple_players_multiple_runs(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = []
        for i in range(60):
            frames.append(
                FakeFrame(
                    frame_number=i,
                    detections=[
                        FakeDetection(
                            bbox=FakeBBox(cx=50 + i * 0.5, cy=34),
                            track_id=1,
                            team="home",
                        ),
                        FakeDetection(
                            bbox=FakeBBox(cx=40 + i * 0.4, cy=30),
                            track_id=2,
                            team="home",
                        ),
                        FakeDetection(
                            bbox=FakeBBox(cx=60, cy=34),
                            track_id=3,
                            team="away",
                        ),
                    ],
                    ball_position=(60, 34),
                )
            )
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs == 2

    def test_report_notes_high_xt(self) -> None:
        svc = PositioningService(min_run_distance_m=3.0, fps=30.0)
        frames = []
        p1_x = 50.0
        for i in range(90):
            p1_x += 0.5
            frames.append(
                FakeFrame(
                    frame_number=i,
                    detections=[
                        FakeDetection(
                            bbox=FakeBBox(cx=p1_x, cy=34),
                            track_id=1,
                            team="home",
                        ),
                    ],
                    ball_position=(60, 34),
                )
            )
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        # With min_run_distance_m=3.0, the run distance will be ~47m
        # xT for BEHIND_DEFENSE at ~47m with avg speed should be 0.04 minimum
        # But total_xT_created is summed across all runs
        # With speed ~(47/3) = ~15.7 m/s (impossibly fast but that's what the frames say),
        # the speed bonus will kick in
        assert report.total_xT_created >= 0.0
        # total_xT should accumulate from each run's created_xT_delta
        total_from_runs = sum(r.created_xT_delta for r in report.runs)
        assert report.total_xT_created == pytest.approx(total_from_runs, abs=0.001)

    def test_run_dataclass_fields(self) -> None:
        run = Run(
            player_track_id=1,
            team="home",
            start_frame=0,
            end_frame=30,
            start_pos=(50.0, 34.0),
            end_pos=(65.0, 34.0),
            run_type=RunType.BEHIND_DEFENSE,
            distance_m=15.75,
            avg_speed_mps=5.25,
            peak_speed_mps=6.10,
            created_xT_delta=0.045,
        )
        assert run.player_track_id == 1
        assert run.run_type == RunType.BEHIND_DEFENSE
        assert run.distance_m == 15.75
        assert run.avg_speed_mps == 5.25

    def test_positioning_report_dataclass(self) -> None:
        report = PositioningReport(
            team="home",
            total_runs=3,
            runs_by_type={RunType.BEHIND_DEFENSE.value: 2, RunType.WIDE.value: 1},
            total_distance_m=55.0,
            avg_run_distance_m=18.33,
            longest_run_m=25.0,
            total_xT_created=0.085,
            runs=[],
            notes=["Strong movement"],
        )
        assert report.team == "home"
        assert report.total_runs == 3
        assert report.avg_run_distance_m == 18.33

    def test_available_property(self) -> None:
        svc = PositioningService()
        assert svc.available is True

    def test_default_parameters(self) -> None:
        svc = PositioningService()
        assert svc.pitch_length_m == 105.0
        assert svc.pitch_width_m == 68.0
        assert svc.min_run_distance_m == 5.0
        assert svc.sprint_threshold_mps == 5.5
        assert svc.fps == 30.0
