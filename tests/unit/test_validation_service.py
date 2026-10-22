"""Tests for ValidationService - accuracy validation against ground truth."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.storage_service import StorageService
from kawkab.services.validation_service import EventGroundTruth, ValidationResult, ValidationService


class TestValidationService:
    """Test accuracy validation utilities."""

    def test_load_ground_truth_json(self):
        svc = ValidationService()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gt.json"
            data = [
                {"event_type": "pass", "timestamp": 45.2, "team": "home", "player_id": 7},
                {"event_type": "shot", "timestamp": 120.5, "team": "away"},
            ]
            path.write_text(json.dumps(data))
            events = svc.load_ground_truth_events(path)
            assert len(events) == 2
            assert events[0].event_type == "pass"
            assert events[0].timestamp == 45.2
            assert events[1].event_type == "shot"

    def test_load_ground_truth_csv(self):
        svc = ValidationService()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gt.csv"
            path.write_text(
                "event_type,timestamp,team,player_id\npass,45.2,home,7\nshot,120.5,away,\n"
            )
            events = svc.load_ground_truth_events(path)
            assert len(events) == 2
            assert events[0].event_type == "pass"
            assert events[0].player_id == 7

    def test_validate_events_perfect_match(self):
        svc = ValidationService()
        gt = [
            EventGroundTruth("pass", 10.0, "home"),
            EventGroundTruth("pass", 20.0, "home"),
            EventGroundTruth("shot", 30.0, "away"),
        ]
        computed = [
            {"type": "pass", "timestamp": 10.0, "team": "home"},
            {"type": "pass", "timestamp": 20.0, "team": "home"},
            {"type": "shot", "timestamp": 30.0, "team": "away"},
        ]
        results = svc.validate_events(computed, gt, tolerance_seconds=2.0)
        f1_results = [r for r in results if r.metric_name == "pass_f1"]
        assert len(f1_results) == 1
        assert f1_results[0].accuracy_score == 1.0
        assert f1_results[0].sample_count == 2

    def test_validate_events_partial_match(self):
        svc = ValidationService()
        gt = [
            EventGroundTruth("pass", 10.0, "home"),
            EventGroundTruth("pass", 20.0, "home"),
        ]
        computed = [
            {"type": "pass", "timestamp": 10.0, "team": "home"},
            {"type": "pass", "timestamp": 25.0, "team": "home"},  # outside tolerance
        ]
        results = svc.validate_events(computed, gt, tolerance_seconds=2.0)
        f1_results = [r for r in results if r.metric_name == "pass_f1"]
        assert f1_results[0].accuracy_score < 1.0
        assert f1_results[0].accuracy_score > 0.0

    def test_validate_possession(self):
        svc = ValidationService()
        result = svc.validate_possession(55.0, 60.0)
        assert result.category == "possession"
        assert result.absolute_error == 5.0
        assert result.accuracy_score == 0.95

    def test_validate_team_assignment(self):
        svc = ValidationService()
        computed = {1: "home", 2: "away", 3: "home", 4: "away"}
        ground_truth = {1: "home", 2: "away", 3: "away", 4: "away"}
        result = svc.validate_team_assignment(computed, ground_truth)
        assert result.accuracy_score == 0.75

    def test_validate_speeds(self):
        svc = ValidationService()
        computed = {1: 32.0, 2: 28.0, 3: 35.0}
        ground_truth = {1: 30.0, 2: 30.0, 3: 33.0}
        results = svc.validate_speeds(computed, ground_truth, max_acceptable_error_kmh=5.0)
        mae_result = [r for r in results if r.metric_name == "max_speed_mae_kmh"][0]
        assert mae_result.computed_value == pytest.approx(2.0, abs=0.01)
        assert mae_result.accuracy_score == pytest.approx(0.6, abs=0.01)

    def test_build_report(self):
        svc = ValidationService()
        results = [
            ValidationResult("events", "pass_f1", 0.8, 1.0, 0.2, 20.0, 0.8, 10),
            ValidationResult("events", "shot_f1", 0.6, 1.0, 0.4, 40.0, 0.6, 5),
        ]
        report = svc.build_report(42, "test", results)
        assert report.match_id == 42
        assert report.ground_truth_source == "test"
        assert report.overall_accuracy == 0.7
        assert "events" in report.summary

    @pytest.mark.asyncio
    async def test_save_validation_to_database(self):
        old_url = os.environ.pop("KAWKAB_DB_URL", None)
        try:
            tmpdir_obj = tempfile.TemporaryDirectory()
            try:
                db_path = Path(tmpdir_obj.name) / "test.db"
                storage = StorageService()
                storage._db_path = db_path
                await storage.initialize()

                # Insert a match first (FK constraint on validation_results)
                conn = storage._conn
                conn.execute(
                    "INSERT INTO matches (id, name, video_path, home_team, away_team, match_date) VALUES (1, 'Test Match', 'test.mp4', 'Home', 'Away', '2025-01-01')"
                )
                conn.commit()

                svc = ValidationService()
                results = [
                    ValidationResult("events", "pass_f1", 0.8, 1.0, 0.2, 20.0, 0.8, 10),
                ]
                report = svc.build_report(1, "manual", results)
                ids = await storage.save_validation_result(report)
                assert len(ids) == 1
                assert ids[0] > 0

                retrieved = await storage.get_validation_results(1)
                assert len(retrieved) == 1
                assert retrieved[0]["metric_name"] == "pass_f1"
                assert retrieved[0]["computed_value"] == 0.8
            finally:
                await storage.close()
                tmpdir_obj.cleanup()
        finally:
            if old_url is not None:
                os.environ["KAWKAB_DB_URL"] = old_url
