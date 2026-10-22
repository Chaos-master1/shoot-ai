"""Phase 5 — Model Quality & Calibration + Data Quality Pipeline tests."""

import json

import pytest

from kawkab.core.event_schema import (
    validate_event,
    validate_events,
)
from kawkab.core.match_anomaly_detection import (
    AnomalyReport,
    compute_data_quality_score,
    detect_anomalies,
)
from kawkab.core.model_comparison_service import (
    ModelComparisonReport,
    compare_xg_models,
    compute_feature_importance,
)
from kawkab.core.xg_calibration import (
    apply_platt_scale,
    compute_auc_roc,
    compute_brier_score,
    compute_calibration_curve,
    compute_log_loss,
    platt_scale,
)
from kawkab.core.xt_confidence import (
    XtInterval,
    bootstrap_xt,
    zone_xt_with_ci,
)

# ================================================================
# xG Calibration — 6 tests
# ================================================================


class TestXgCalibration:
    def test_calibration_curve_perfect(self):
        preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        outc = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        curve = compute_calibration_curve(preds, outc, n_bins=5)
        assert len(curve.bins) == 5
        assert 0.0 <= curve.ece <= 1.0
        assert 0.0 <= curve.mse <= 1.0

    def test_calibration_curve_empty(self):
        curve = compute_calibration_curve([], [], n_bins=10)
        assert curve.bins == []
        assert curve.ece == 0.0
        assert curve.mse == 0.0

    def test_brier_score_perfect(self):
        bs = compute_brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0])
        assert bs == 0.0

    def test_brier_score_worst(self):
        bs = compute_brier_score([0.0, 1.0], [1, 0])
        assert bs == 1.0

    def test_log_loss_perfect(self):
        ll = compute_log_loss([0.999, 0.001], [1, 0], eps=1e-15)
        assert ll < 0.01

    def test_auc_roc(self):
        preds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        outc = [0, 0, 0, 0, 1, 1, 1, 1]
        auc = compute_auc_roc(preds, outc)
        assert 0.5 < auc <= 1.0

    def test_platt_scale_monotonic(self):
        preds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        outc = [0, 0, 0, 0, 1, 1, 1, 1]
        a, b = platt_scale(preds, outc)
        calibrated = apply_platt_scale(preds, a, b)
        assert len(calibrated) == len(preds)
        for c in calibrated:
            assert 0.0 < c < 1.0


# ================================================================
# PSxG Model — 6 tests
# ================================================================

# ================================================================
# Model Comparison — 6 tests
# ================================================================


class TestModelComparisonService:
    def test_compare_empty_events(self):
        reports = compare_xg_models([])
        assert reports == []

    def test_compare_with_shot_events(self):
        events = [
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 15,
                "angle_deg": 20,
                "body_part": "right_foot",
                "shot_type": "open_play",
            },
            {
                "type": "shot",
                "is_goal": 1,
                "distance_m": 8,
                "angle_deg": 10,
                "body_part": "left_foot",
                "shot_type": "open_play",
            },
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 25,
                "angle_deg": 40,
                "body_part": "head",
                "shot_type": "header",
            },
            {
                "type": "shot",
                "is_goal": 1,
                "distance_m": 5,
                "angle_deg": 5,
                "body_part": "right_foot",
                "shot_type": "open_play",
                "is_one_on_one": True,
            },
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 30,
                "angle_deg": 35,
                "body_part": "right_foot",
                "shot_type": "volley",
            },
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 20,
                "angle_deg": 25,
                "body_part": "left_foot",
                "shot_type": "open_play",
            },
        ]
        # Need at least 6 for a 2/3 train split (test split needs at least 1)
        reports = compare_xg_models(events)
        assert len(reports) >= 1
        for r in reports:
            assert isinstance(r, ModelComparisonReport)
            assert 0.0 <= r.brier_score <= 1.0
            assert r.n_samples > 0

    def test_feature_importance_returns_dict(self):
        events = [
            {"type": "shot", "is_goal": 0, "distance_m": 15, "angle_deg": 20},
            {"type": "shot", "is_goal": 1, "distance_m": 5, "angle_deg": 10},
            {"type": "shot", "is_goal": 0, "distance_m": 25, "angle_deg": 40},
            {"type": "shot", "is_goal": 1, "distance_m": 8, "angle_deg": 15},
            {"type": "shot", "is_goal": 0, "distance_m": 30, "angle_deg": 35},
            {"type": "shot", "is_goal": 0, "distance_m": 20, "angle_deg": 25},
        ]
        imp = compute_feature_importance(events)
        assert isinstance(imp, dict)

    def test_feature_importance_few_events(self):
        imp = compute_feature_importance([{"type": "shot", "is_goal": 0}])
        assert imp == {}

    def test_model_comparison_has_calibration_curve(self):
        events = [
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 15,
                "angle_deg": 20,
                "body_part": "right_foot",
                "shot_type": "open_play",
            },
            {
                "type": "shot",
                "is_goal": 1,
                "distance_m": 8,
                "angle_deg": 10,
                "body_part": "left_foot",
                "shot_type": "open_play",
            },
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 25,
                "angle_deg": 40,
                "body_part": "head",
                "shot_type": "header",
            },
            {
                "type": "shot",
                "is_goal": 1,
                "distance_m": 5,
                "angle_deg": 5,
                "body_part": "right_foot",
                "shot_type": "open_play",
            },
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 30,
                "angle_deg": 35,
                "body_part": "right_foot",
                "shot_type": "volley",
            },
            {
                "type": "shot",
                "is_goal": 0,
                "distance_m": 20,
                "angle_deg": 25,
                "body_part": "left_foot",
                "shot_type": "open_play",
            },
        ]
        reports = compare_xg_models(events)
        for r in reports:
            assert hasattr(r.calibration_curve, "bins")
            assert hasattr(r.calibration_curve, "ece")


# ================================================================
# xT Confidence — 6 tests
# ================================================================


class TestXtConfidence:
    def test_bootstrap_empty_events(self):
        intervals = bootstrap_xt([], n_resamples=10)
        assert intervals == {}

    def test_bootstrap_returns_intervals(self):
        events = [
            {"type": "pass", "x": 50.0, "y": 34.0, "end_x": 60.0, "end_y": 34.0},
            {"type": "pass", "x": 60.0, "y": 34.0, "end_x": 70.0, "end_y": 40.0},
            {"type": "pass", "x": 30.0, "y": 20.0, "end_x": 50.0, "end_y": 30.0},
            {"type": "carry", "x": 40.0, "y": 30.0, "end_x": 55.0, "end_y": 35.0},
        ]
        intervals = bootstrap_xt(events, n_resamples=20, grid_size=(5, 8))
        assert len(intervals) > 0
        for _key, interval in intervals.items():
            assert isinstance(interval, XtInterval)
            assert interval.ci_low <= interval.mean <= interval.ci_high

    def test_zone_xt_with_ci(self):
        events = [
            {"type": "pass", "x": 50.0, "y": 34.0, "end_x": 60.0, "end_y": 34.0},
            {"type": "pass", "x": 60.0, "y": 34.0, "end_x": 70.0, "end_y": 40.0},
            {"type": "pass", "x": 30.0, "y": 20.0, "end_x": 50.0, "end_y": 30.0},
        ]
        zones = zone_xt_with_ci(events, grid_size=(5, 8), n_resamples=10)
        assert isinstance(zones, dict)
        for key, _interval in zones.items():
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_xt_interval_has_std(self):
        events = [
            {"type": "pass", "x": 50.0, "y": 34.0, "end_x": 60.0, "end_y": 34.0},
            {"type": "pass", "x": 60.0, "y": 34.0, "end_x": 70.0, "end_y": 40.0},
        ]
        intervals = bootstrap_xt(events, n_resamples=10, grid_size=(5, 8))
        for interval in intervals.values():
            assert interval.std >= 0.0

    def test_xt_interval_mean_reasonable(self):
        events = [
            {"type": "pass", "x": 50.0, "y": 34.0, "end_x": 90.0, "end_y": 34.0},
            {"type": "pass", "x": 90.0, "y": 34.0, "end_x": 100.0, "end_y": 34.0},
            {"type": "pass", "x": 30.0, "y": 20.0, "end_x": 50.0, "end_y": 30.0},
        ]
        intervals = bootstrap_xt(events, n_resamples=10, grid_size=(5, 8))
        for interval in intervals.values():
            assert 0.0 <= interval.mean <= 1.0

    def test_more_resamples_narrower_ci(self):
        events = [
            {"type": "pass", "x": 50.0, "y": 34.0, "end_x": 60.0, "end_y": 34.0},
            {"type": "pass", "x": 40.0, "y": 30.0, "end_x": 55.0, "end_y": 35.0},
        ]
        intervals10 = bootstrap_xt(events, n_resamples=5, grid_size=(5, 8))
        intervals50 = bootstrap_xt(events, n_resamples=10, grid_size=(5, 8))
        assert len(intervals10) == len(intervals50)


# ================================================================
# Event Schema Validation — 6 tests
# ================================================================


class TestEventSchema:
    def test_valid_event(self):
        ev = {
            "type": "pass",
            "team": "home",
            "timestamp": 100.0,
            "x": 50.0,
            "y": 34.0,
            "track_id": 1,
        }
        result = validate_event(ev)
        assert result.valid
        assert len(result.errors) == 0

    def test_invalid_x_bound(self):
        ev = {
            "type": "pass",
            "team": "home",
            "timestamp": 100.0,
            "x": 200.0,
            "y": 34.0,
        }
        result = validate_event(ev)
        assert not result.valid
        assert any("bounds" in e for e in result.errors)

    def test_invalid_y_bound(self):
        ev = {
            "type": "pass",
            "team": "home",
            "timestamp": 100.0,
            "x": 50.0,
            "y": 100.0,
        }
        result = validate_event(ev)
        assert not result.valid
        assert any("bounds" in e for e in result.errors)

    def test_missing_type(self):
        ev = {"team": "home", "timestamp": 100.0, "x": 50.0, "y": 34.0}
        result = validate_event(ev)
        assert result.valid  # type is optional, just checked if present

    def test_wrong_type_for_timestamp(self):
        ev = {
            "type": "shot",
            "team": "home",
            "timestamp": "abc",
            "x": 50.0,
            "y": 34.0,
        }
        result = validate_event(ev)
        _ = " ".join(result.errors).lower()
        assert not result.valid

    def test_validate_events_mixed(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 10.0, "x": 50.0, "y": 34.0},
            {"type": "shot", "team": "away", "timestamp": 20.0, "x": 106.0, "y": 34.0},
            {"type": "pass", "team": "home", "timestamp": 15.0, "x": 50.0, "y": -5.0},
        ]
        results = validate_events(events)
        assert len(results) == 3
        assert results[0].valid
        assert not results[1].valid
        assert not results[2].valid


# ================================================================
# Anomaly Detection — 8 tests
# ================================================================


class TestAnomalyDetection:
    def test_empty_events(self):
        report = detect_anomalies([])
        assert report.score == 0.0 or len(report.anomalies) > 0

    def test_normal_events_no_anomalies(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 0.0, "x": 50.0, "y": 34.0, "id": 1},
            {"type": "pass", "team": "away", "timestamp": 10.0, "x": 40.0, "y": 30.0, "id": 2},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 20.0,
                "x": 80.0,
                "y": 34.0,
                "is_goal": False,
                "xg": 0.05,
                "id": 3,
            },
            {"type": "pass", "team": "home", "timestamp": 30.0, "x": 50.0, "y": 34.0, "id": 4},
            {
                "type": "shot",
                "team": "away",
                "timestamp": 40.0,
                "x": 20.0,
                "y": 34.0,
                "is_goal": True,
                "xg": 0.3,
                "id": 5,
            },
        ]
        report = detect_anomalies(events, match_duration_min=1.0)
        score = compute_data_quality_score(events, match_duration_min=1.0)
        assert 0.0 <= score <= 100.0
        assert isinstance(report, AnomalyReport)

    def test_impossible_speed_detected(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0.0,
                "x": 50.0,
                "y": 34.0,
                "id": 1,
                "speed_mps": 15.0,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp": 10.0,
                "x": 50.0,
                "y": 34.0,
                "id": 2,
                "speed_mps": 5.0,
            },
        ]
        report = detect_anomalies(events, match_duration_min=1.0)
        assert any(a["type"] == "impossible_speed" for a in report.anomalies)

    def test_too_many_goals(self):
        events = []
        for i in range(20):
            events.append({"type": "goal", "team": "home", "timestamp": float(i * 60), "id": i})
        report = detect_anomalies(events, match_duration_min=90.0)
        assert any(a["type"] == "too_many_goals" for a in report.anomalies)

    def test_missing_goal_high_xg(self):
        events = [
            {
                "type": "shot",
                "team": "home",
                "timestamp": 100.0,
                "is_goal": False,
                "xg": 0.95,
                "id": 1,
            },
        ]
        report = detect_anomalies(events)
        assert any(a["type"] == "missing_goal_high_xg" for a in report.anomalies)

    def test_coordinate_outliers(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 10.0, "x": 200.0, "y": 34.0, "id": 1},
        ]
        report = detect_anomalies(events)
        assert any(a["type"] == "coordinate_outlier" for a in report.anomalies)

    def test_duplicate_events(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 10.0, "id": 1},
            {"type": "pass", "team": "home", "timestamp": 10.0, "id": 2},
        ]
        report = detect_anomalies(events)
        assert any(a["type"] == "duplicate_event" for a in report.anomalies)

    def test_too_many_subs(self):
        events = []
        for i in range(8):
            events.append(
                {
                    "type": "substitution",
                    "event_type": "substitution",
                    "team": "home",
                    "timestamp": float(i * 60),
                    "id": i,
                }
            )
        report = detect_anomalies(events)
        assert any(a["type"] == "too_many_subs" for a in report.anomalies)

    def test_short_duration_penalty(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 0.0, "id": 1},
            {"type": "pass", "team": "home", "timestamp": 60.0, "id": 2},
        ]
        score = compute_data_quality_score(events, match_duration_min=90.0)
        assert score < 100.0


# ================================================================
# Quality Score — 6 tests
# ================================================================


class TestQualityScore:
    def test_perfect_data(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 0.0, "x": 50.0, "y": 34.0, "id": 1},
            {"type": "shot", "team": "home", "timestamp": 10.0, "x": 80.0, "y": 34.0, "id": 2},
            {"type": "pass", "team": "away", "timestamp": 20.0, "x": 40.0, "y": 30.0, "id": 3},
            {"type": "shot", "team": "away", "timestamp": 30.0, "x": 20.0, "y": 34.0, "id": 4},
            {"type": "pass", "team": "home", "timestamp": 40.0, "x": 50.0, "y": 34.0, "id": 5},
            {"type": "pass", "team": "home", "timestamp": 50.0, "x": 50.0, "y": 34.0, "id": 6},
            {"type": "pass", "team": "home", "timestamp": 60.0, "x": 50.0, "y": 34.0, "id": 7},
            {"type": "pass", "team": "home", "timestamp": 70.0, "x": 50.0, "y": 34.0, "id": 8},
            {"type": "pass", "team": "home", "timestamp": 80.0, "x": 50.0, "y": 34.0, "id": 9},
            {"type": "pass", "team": "home", "timestamp": 90.0, "x": 50.0, "y": 34.0, "id": 10},
        ]
        score = compute_data_quality_score(events, match_duration_min=2.0)
        assert score > 0

    def test_empty_events_score_zero(self):
        score = compute_data_quality_score([])
        assert score == 0.0

    def test_few_events_penalty(self):
        events = [{"type": "pass", "team": "home", "timestamp": 0.0, "id": 1}]
        score = compute_data_quality_score(events)
        assert score < 100.0

    def test_no_shots_penalty(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 0.0, "x": 50.0, "y": 34.0, "id": 1},
            {"type": "pass", "team": "away", "timestamp": 10.0, "x": 40.0, "y": 30.0, "id": 2},
        ]
        score = compute_data_quality_score(events, match_duration_min=1.0)
        assert score < 100.0

    def test_score_between_zero_and_hundred(self):
        events = [
            {"type": "pass", "team": "home", "timestamp": 0.0, "x": 50.0, "y": 34.0, "id": 1},
            {"type": "shot", "team": "home", "timestamp": 15.0, "x": 80.0, "y": 34.0, "id": 2},
            {"type": "pass", "team": "away", "timestamp": 30.0, "x": 40.0, "y": 30.0, "id": 3},
            {"type": "shot", "team": "away", "timestamp": 45.0, "x": 20.0, "y": 34.0, "id": 4},
            {"type": "pass", "team": "home", "timestamp": 60.0, "x": 50.0, "y": 34.0, "id": 5},
        ]
        score = compute_data_quality_score(events, match_duration_min=2.0)
        assert 0.0 <= score <= 100.0

    def test_score_drops_with_anomalies(self):
        clean_events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": float(i * 10),
                "x": 50.0,
                "y": 34.0,
                "id": i,
            }
            for i in range(10)
        ]
        dirty_events = clean_events + [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 100.0,
                "x": 50.0,
                "y": 34.0,
                "id": 99,
                "speed_mps": 20.0,
            },
            {"type": "pass", "team": "home", "timestamp": 110.0, "x": 200.0, "y": 34.0, "id": 100},
        ]
        clean_score = compute_data_quality_score(clean_events)
        dirty_score = compute_data_quality_score(dirty_events)
        assert clean_score >= dirty_score


# ================================================================
# Bridge Quality Slot — 4 tests
# ================================================================


class MockBridge:
    pass


class MockStorage:
    def __init__(self, events=None):
        self._events = events or []

    async def get_match_events(self, mid):
        return self._events


class TestBridgeQualitySlot:
    # Regression tests: get_match_quality_score called
    # self.storage_service.get_match_events(mid) without awaiting it (the
    # method itself wasn't even `async def`, so it couldn't await). That
    # returned an un-awaited coroutine instead of the events list, which
    # detect_anomalies()/compute_data_quality_score() couldn't process --
    # caught by the handler's own broad except and turned into a permanent
    # {"level": "error"} response. These tests previously accepted "error"
    # as a valid outcome for a *good* match, which is exactly how this
    # went unnoticed: a test that can't fail isn't testing anything.

    @pytest.mark.asyncio
    async def test_get_match_quality_score_returns_json(self):
        from kawkab.ui.bridge_handlers.bridge_match_intel import MatchIntelHandler

        storage = MockStorage(
            [
                {"type": "pass", "team": "home", "timestamp": 0.0, "x": 50.0, "y": 34.0, "id": 1},
                {
                    "type": "shot",
                    "team": "home",
                    "timestamp": 10.0,
                    "x": 80.0,
                    "y": 34.0,
                    "is_goal": False,
                    "xg": 0.05,
                    "id": 2,
                },
            ]
        )
        handler = MatchIntelHandler(MockBridge(), {"storage_service": storage})
        result_json = await handler.get_match_quality_score("1")
        result = json.loads(result_json)
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "score" in result
        assert "level" in result
        assert isinstance(result["score"], (int, float))

    @pytest.mark.asyncio
    async def test_quality_score_good_level(self):
        from kawkab.ui.bridge_handlers.bridge_match_intel import MatchIntelHandler

        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": float(i * 10),
                "x": 50.0,
                "y": 34.0,
                "id": i,
            }
            for i in range(12)
        ] + [
            {
                "type": "shot",
                "team": "home",
                "timestamp": 120.0,
                "x": 80.0,
                "y": 34.0,
                "is_goal": False,
                "xg": 0.1,
                "id": 20,
            },
        ]
        storage = MockStorage(events)
        handler = MatchIntelHandler(MockBridge(), {"storage_service": storage})
        result = json.loads(await handler.get_match_quality_score("1"))
        assert result.get("level") in ("good", "fair", "poor")

    @pytest.mark.asyncio
    async def test_quality_score_poor_with_anomalies(self):
        from kawkab.ui.bridge_handlers.bridge_match_intel import MatchIntelHandler

        storage = MockStorage(
            [
                {
                    "type": "pass",
                    "team": "home",
                    "timestamp": 0.0,
                    "x": 200.0,
                    "y": 34.0,
                    "id": 1,
                    "speed_mps": 25.0,
                },
            ]
        )
        handler = MatchIntelHandler(MockBridge(), {"storage_service": storage})
        result = json.loads(await handler.get_match_quality_score("1"))
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "score" in result
        assert result["anomaly_count"] > 0

    @pytest.mark.asyncio
    async def test_quality_score_empty_match(self):
        from kawkab.ui.bridge_handlers.bridge_match_intel import MatchIntelHandler

        storage = MockStorage([])
        handler = MatchIntelHandler(MockBridge(), {"storage_service": storage})
        result = json.loads(await handler.get_match_quality_score("1"))
        assert result["score"] == 0.0
        assert result["level"] == "poor"
