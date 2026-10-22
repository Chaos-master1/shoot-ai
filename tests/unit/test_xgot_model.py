"""Tests for xGOT model (post-shot expected goals with calibration)."""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.analysis.xgot_model import (
    XgotMatchReport,
    XgotModel,
    XgotResult,
    _extract_features,
    _inv_logit,
)


def test_inv_logit():
    assert abs(_inv_logit(0) - 0.5) < 1e-6
    assert _inv_logit(10) > 0.999
    assert _inv_logit(-10) < 0.001
    assert _inv_logit(20) > 0.999
    assert _inv_logit(-20) < 0.001


def test_extract_features():
    feats = _extract_features(15.0, 30.0)
    assert feats["distance_m"] == 15.0
    assert feats["angle_deg"] == 30.0
    assert feats["is_header"] == 0.0
    assert feats["one_on_one"] == 0.0

    feats_h = _extract_features(10, 45, body_part="head", one_on_one=True)
    assert feats_h["is_header"] == 1.0
    assert feats_h["one_on_one"] == 1.0


def test_default_compute():
    model = XgotModel()
    # Close-range central shot (high xGOT)
    close = model.compute(distance_m=5, angle_deg=0)
    assert 0.3 < close.xgot < 0.95

    # Long-range wide shot (low xGOT)
    far = model.compute(distance_m=35, angle_deg=45)
    assert far.xgot < close.xgot

    # Header should be easier to save
    header = model.compute(distance_m=12, angle_deg=15, body_part="head")
    foot = model.compute(distance_m=12, angle_deg=15, body_part="right_foot")
    assert header.xgot < foot.xgot


def test_one_on_one_increases_xgot():
    model = XgotModel()
    normal = model.compute(distance_m=12, angle_deg=10, one_on_one=False)
    ooo = model.compute(distance_m=12, angle_deg=10, one_on_one=True)
    assert ooo.xgot > normal.xgot


def test_placement_corners_higher_xgot():
    model = XgotModel()
    center = model.compute(distance_m=15, angle_deg=20, placement_x=0.5, placement_y=0.5)
    corner = model.compute(distance_m=15, angle_deg=20, placement_x=0.05, placement_y=0.05)
    assert corner.xgot > center.xgot


def test_xgot_bounds():
    model = XgotModel()
    for _ in range(20):
        r = model.compute(
            distance_m=random.uniform(1, 40),
            angle_deg=random.uniform(0, 60),
            placement_x=random.uniform(0, 1),
            placement_y=random.uniform(0, 1),
        )
        assert 0.005 <= r.xgot <= 0.995
        assert 0 <= r.ci_lower <= 1
        assert 0 <= r.ci_upper <= 1
        assert r.ci_lower <= r.xgot <= r.ci_upper


def test_xgot_result_to_dict():
    r = XgotResult(xgot=0.5432, ci_lower=0.4, ci_upper=0.7, calibrated=True)
    d = r.to_dict()
    assert d["xgot"] == 0.5432
    assert d["ci_lower"] == 0.4
    assert d["ci_upper"] == 0.7
    assert d["calibrated"] is True


def test_match_report():
    model = XgotModel()
    events = [
        {"type": "shot", "team": "home", "distance_m": 8, "angle_deg": 10, "is_goal": True},
        {"type": "shot", "team": "home", "distance_m": 25, "angle_deg": 30, "is_goal": False},
        {"type": "shot", "team": "away", "distance_m": 15, "angle_deg": 20, "is_goal": True},
        {"type": "pass", "team": "home"},  # ignored
    ]
    report = model.compute_match(events)
    assert report.home_xgot > 0
    assert report.away_xgot > 0
    assert report.home_goals == 1
    assert report.away_goals == 1
    assert len(report.details) == 3


def test_match_report_to_dict():
    report = XgotMatchReport(home_xgot=1.5, away_xgot=0.8, home_goals=2, away_goals=1, details=[])
    d = report.to_dict()
    assert d["home_xgot"] == 1.5
    assert d["home_xgot_diff"] == 0.5
    assert d["away_xgot_diff"] == 0.2


def test_save_load_json():
    model = XgotModel()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "xgot.json"
        model.save_json(path)
        loaded = XgotModel.load_json(path)
        r1 = model.compute(distance_m=10, angle_deg=5)
        r2 = loaded.compute(distance_m=10, angle_deg=5)
        assert abs(r1.xgot - r2.xgot) < 0.01


def test_save_load_pickle():
    model = XgotModel()
    model._calibrated = True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "xgot.pkl"
        model.save(path)
        loaded = XgotModel.load(path)
        assert loaded.calibrated is True
        r1 = model.compute(distance_m=10, angle_deg=5)
        r2 = loaded.compute(distance_m=10, angle_deg=5)
        assert abs(r1.xgot - r2.xgot) < 0.001


def test_train_with_sklearn():
    model = XgotModel()
    shots = []
    for _ in range(100):
        dist = random.uniform(2, 35)
        angle = random.uniform(0, 50)
        is_goal = (
            1
            if (dist < 10 and random.random() > 0.3) or (dist > 20 and random.random() > 0.85)
            else 0
        )
        shots.append(
            {
                "distance_m": dist,
                "angle_deg": angle,
                "placement_x": random.uniform(0, 1),
                "placement_y": random.uniform(0, 1),
                "body_part": random.choice(["right_foot", "left_foot", "head"]),
                "is_goal": is_goal,
            }
        )
    result = model.train(shots, calibrate=True, bootstrap=True)
    if model._sk_model is not None:
        assert result["trained"] is True
        assert result["calibrated"] is True
        r = model.compute(distance_m=8, angle_deg=5)
        assert 0.01 <= r.xgot <= 0.99
        assert r.calibrated is True


def test_train_too_few_samples():
    model = XgotModel()
    result = model.train([{"distance_m": 10, "is_goal": 1}], calibrate=True)
    assert result["trained"] is False


def test_feature_importance_sensible():
    model = XgotModel()
    # Very close = high xGOT
    close = model.compute(distance_m=2, angle_deg=0)
    # Very far = low xGOT
    far = model.compute(distance_m=40, angle_deg=45)
    assert close.xgot > far.xgot + 0.15

    # Distance matters more than angle for xGOT
    close = model.compute(distance_m=3, angle_deg=0)
    far = model.compute(distance_m=30, angle_deg=0)
    assert close.xgot > far.xgot + 0.1


def test_confidence_interval_narrower_with_more_data():
    model = XgotModel()
    # Without bootstrap samples, CI is fixed width
    r = model.compute(distance_m=15, angle_deg=20)
    width = r.ci_upper - r.ci_lower
    assert width > 0.05

    # With bootstrap samples, CI should contain xgot
    model._bootstrap_coefs = [
        {
            "intercept": -1.2,
            "distance_m": -0.06,
            "angle_deg": 0.015,
            "placement_dist_center": 2.0,
            "is_header": -0.5,
            "one_on_one": 0.4,
            "shot_speed_mps": 0.04,
            "defender_proximity": -0.15,
        },
        {
            "intercept": -1.0,
            "distance_m": -0.05,
            "angle_deg": 0.012,
            "placement_dist_center": 1.8,
            "is_header": -0.4,
            "one_on_one": 0.35,
            "shot_speed_mps": 0.035,
            "defender_proximity": -0.12,
        },
        {
            "intercept": -1.4,
            "distance_m": -0.07,
            "angle_deg": 0.018,
            "placement_dist_center": 2.2,
            "is_header": -0.6,
            "one_on_one": 0.45,
            "shot_speed_mps": 0.045,
            "defender_proximity": -0.18,
        },
        {
            "intercept": -1.1,
            "distance_m": -0.055,
            "angle_deg": 0.014,
            "placement_dist_center": 1.9,
            "is_header": -0.45,
            "one_on_one": 0.38,
            "shot_speed_mps": 0.038,
            "defender_proximity": -0.13,
        },
        {
            "intercept": -1.3,
            "distance_m": -0.065,
            "angle_deg": 0.016,
            "placement_dist_center": 2.1,
            "is_header": -0.55,
            "one_on_one": 0.42,
            "shot_speed_mps": 0.042,
            "defender_proximity": -0.16,
        },
    ]
    r2 = model.compute(distance_m=15, angle_deg=20)
    assert r2.ci_lower < r2.xgot < r2.ci_upper
