"""Regression tests comparing Kawkab xG models against StatsBomb ground truth.

Uses real match data from data/ground_truth/statsbomb/events/.
Fails if models regress beyond tolerance thresholds."""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import pytest

from kawkab.core.dl_xg_model import predict_dl_xg
from kawkab.core.xg_model import compute_xg

GT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ground_truth" / "statsbomb"
EVENT_DIR = GT_DIR / "events"

# data/statsbomb_corpus/ is committed to the repo and carries the same
# StatsBomb open-data event schema. Use it as a hermetic fallback whenever
# the network-fetched ground truth is missing or too thin: CI has seen the
# conftest auto-fetch partially fail (2 of 5 files), which made this gate
# nondeterministically fail on "Need >= 3 matches".
CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "statsbomb_corpus"


def _has_enough_files(d: Path) -> bool:
    return d.is_dir() and len(list(d.glob("*.json"))) >= 3


def _pick_event_dir() -> Path:
    if _has_enough_files(EVENT_DIR):
        return EVENT_DIR
    if _has_enough_files(CORPUS_DIR):
        return CORPUS_DIR
    return EVENT_DIR  # nothing viable; keep original path for the skip reason


EVENTS = _pick_event_dir()

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_CENTER_X = PITCH_LENGTH
GOAL_CENTER_Y = PITCH_WIDTH / 2.0
GOAL_WIDTH = 7.32

TOLERANCE_MAE = 0.12
TOLERANCE_RMSE = 0.18
TOLERANCE_BIAS = 0.05


def _get_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _get_angle(x, y):
    """Subtended goal-OPENING angle (center + half goal width, capped 90).

    NOTE on conventions: the legacy hand-tuned compute_xg coefficients
    were calibrated against this subtended convention (central 6m ~90,
    far wide shots -> small), and this file's tolerances were validated
    against statsbomb_xg with it. The ENHANCED model + trainer use the
    deviation-from-center convention instead (see
    scripts/train_xg_from_statsbomb.py::_get_angle); both are
    internally consistent within their own model. Do not "fix" one to
    match the other without re-validating the tolerances here.
    """
    dx = GOAL_CENTER_X - x
    dy = GOAL_CENTER_Y - y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1.0:
        return 90.0
    half_goal = math.degrees(math.atan2(GOAL_WIDTH / 2, dist))
    angle_to_center = abs(math.degrees(math.atan2(dy, dx)))
    return min(angle_to_center + half_goal, 90.0)


def _map_body_part(sb_part):
    m = {
        "Head": "head",
        "Left Foot": "left_foot",
        "Right Foot": "right_foot",
        "Other": "right_foot",
    }
    return m.get(sb_part, "right_foot")


def _map_shot_type(sb_type):
    m = {
        "Open Play": "open_play",
        "Volley": "volley",
        "Half Volley": "half_volley",
        "Free Kick": "free_kick",
        "Penalty": "penalty",
        "Corner": "open_play",
        "Set Piece": "free_kick",
        "Direct Free Kick": "free_kick",
    }
    return m.get(sb_type, "open_play")


def load_all_shots():
    """Load all shots from all StatsBomb match files (ground truth, with the
    committed corpus as fallback when the fetched ground truth is thin)."""
    shots = []
    for fpath in sorted(glob.glob(str(EVENTS / "*.json"))):
        try:
            events = json.loads(Path(fpath).read_text(encoding="utf-8"))
        except Exception:
            continue
        match_id = int(Path(fpath).stem)
        for ev in events:
            type_info = ev.get("type", {}) or {}
            if type_info.get("name") != "Shot":
                continue
            shot_info = ev.get("shot", {}) or {}
            sb_xg = shot_info.get("statsbomb_xg")
            if sb_xg is None:
                continue
            loc = ev.get("location") or []
            if len(loc) < 2:
                continue
            x, y = float(loc[0]), float(loc[1])
            body_part = _map_body_part((shot_info.get("body_part") or {}).get("name", ""))
            shot_type = _map_shot_type((shot_info.get("type") or {}).get("name", ""))
            is_goal = (shot_info.get("outcome") or {}).get("name") == "Goal"
            shots.append(
                {
                    "match_id": match_id,
                    "x": x,
                    "y": y,
                    "distance_m": _get_distance(x, y, GOAL_CENTER_X, GOAL_CENTER_Y),
                    "angle_deg": _get_angle(x, y),
                    "body_part": body_part,
                    "shot_type": shot_type,
                    "is_goal": is_goal,
                    "statsbomb_xg": float(sb_xg),
                }
            )
    return shots


SHOTS = None


@pytest.fixture(scope="session")
def all_shots():
    global SHOTS
    if SHOTS is None:
        SHOTS = load_all_shots()
    if not SHOTS:
        pytest.skip("No StatsBomb shot data available (ground truth fetch failed and corpus empty)")
    return SHOTS


class TestXgRegression:
    """Regression tests: Kawkab xG models vs StatsBomb ground truth."""

    def test_heuristic_model_mae_within_tolerance(self, all_shots):
        errors = []
        for s in all_shots:
            pred = compute_xg(
                distance_m=s["distance_m"],
                angle_deg=s["angle_deg"],
                body_part=s["body_part"],
                shot_type=s["shot_type"],
            )
            errors.append(abs(pred - s["statsbomb_xg"]))
        mae = sum(errors) / len(errors)
        assert mae < TOLERANCE_MAE, f"Heuristic MAE {mae:.4f} >= {TOLERANCE_MAE}"

    def test_heuristic_model_rmse_within_tolerance(self, all_shots):
        errors = [
            (
                compute_xg(
                    distance_m=s["distance_m"],
                    angle_deg=s["angle_deg"],
                    body_part=s["body_part"],
                    shot_type=s["shot_type"],
                )
                - s["statsbomb_xg"]
            )
            for s in all_shots
        ]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        assert rmse < TOLERANCE_RMSE, f"Heuristic RMSE {rmse:.4f} >= {TOLERANCE_RMSE}"

    def test_heuristic_model_bias_within_tolerance(self, all_shots):
        errors = [
            (
                compute_xg(
                    distance_m=s["distance_m"],
                    angle_deg=s["angle_deg"],
                    body_part=s["body_part"],
                    shot_type=s["shot_type"],
                )
                - s["statsbomb_xg"]
            )
            for s in all_shots
        ]
        bias = sum(errors) / len(errors)
        assert abs(bias) < TOLERANCE_BIAS, f"Heuristic bias {bias:.4f} >= |{TOLERANCE_BIAS}|"

    def test_enhanced_model_mae_within_tolerance(self, all_shots):
        errors = []
        from kawkab.core.xg_model import ENHANCED_COEFFICIENTS, EnhancedXgFeatures, EnhancedXgModel

        em = EnhancedXgModel(coefficients=ENHANCED_COEFFICIENTS)
        for s in all_shots:
            pred = em.compute_single(
                EnhancedXgFeatures(
                    distance_m=s["distance_m"],
                    angle_deg=s["angle_deg"],
                    is_header=(s["body_part"] == "head"),
                    is_volley=(s["shot_type"] in ("volley", "half_volley")),
                    is_free_kick=(s["shot_type"] == "free_kick"),
                    is_penalty=(s["shot_type"] == "penalty"),
                )
            )
            errors.append(abs(pred - s["statsbomb_xg"]))
        mae = sum(errors) / len(errors)
        assert mae < TOLERANCE_MAE + 0.03, f"Enhanced MAE {mae:.4f} >= {TOLERANCE_MAE + 0.03}"

    def test_enhanced_model_rmse_within_tolerance(self, all_shots):
        from kawkab.core.xg_model import ENHANCED_COEFFICIENTS, EnhancedXgFeatures, EnhancedXgModel

        em = EnhancedXgModel(coefficients=ENHANCED_COEFFICIENTS)
        errors = [
            (
                em.compute_single(
                    EnhancedXgFeatures(
                        distance_m=s["distance_m"],
                        angle_deg=s["angle_deg"],
                        is_header=(s["body_part"] == "head"),
                        is_volley=(s["shot_type"] in ("volley", "half_volley")),
                        is_free_kick=(s["shot_type"] == "free_kick"),
                        is_penalty=(s["shot_type"] == "penalty"),
                    )
                )
                - s["statsbomb_xg"]
            )
            for s in all_shots
        ]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        assert rmse < TOLERANCE_RMSE + 0.03, f"Enhanced RMSE {rmse:.4f} >= {TOLERANCE_RMSE + 0.03}"

    def test_dl_model_mae_within_tolerance(self, all_shots):
        errors = []
        for s in all_shots:
            pred = predict_dl_xg(
                distance_m=s["distance_m"],
                angle_deg=s["angle_deg"],
                is_header=(s["body_part"] == "head"),
                shot_type=s["shot_type"],
            )
            errors.append(abs(pred - s["statsbomb_xg"]))
        mae = sum(errors) / len(errors)
        assert mae < TOLERANCE_MAE + 0.05, f"DL MAE {mae:.4f} >= {TOLERANCE_MAE + 0.05}"

    def test_heuristic_distance_monotonic_on_real_data(self, all_shots):
        foot_shots = [
            s
            for s in all_shots
            if s["body_part"] != "head" and s["shot_type"] in ("open_play", "volley")
        ]
        grouped: dict[str, list] = {}
        for s in foot_shots:
            key = f"{s['body_part']}_{s['shot_type']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(s)
        violations = 0
        total = 0
        for _key, group in grouped.items():
            sorted_group = sorted(group, key=lambda x: x["distance_m"])
            prev_xg = 1.0
            for s in sorted_group:
                xg = compute_xg(
                    distance_m=s["distance_m"],
                    angle_deg=s["angle_deg"],
                    body_part=s["body_part"],
                    shot_type=s["shot_type"],
                )
                if xg > prev_xg + 0.02:
                    violations += 1
                prev_xg = xg
                total += 1
        assert violations / max(total, 1) < 0.30, (
            f"Too many monotonicity violations: {violations}/{total}"
        )

    def test_heuristic_angle_monotonic_on_real_data(self, all_shots):
        foot_shots = [
            s
            for s in all_shots
            if s["body_part"] != "head" and s["shot_type"] in ("open_play", "volley")
        ]
        narrow = sorted(
            [s for s in foot_shots if 10 < s["distance_m"] < 20], key=lambda s: s["angle_deg"]
        )
        violations = 0
        total = 0
        if len(narrow) >= 4:
            prev_xg = 1.0
            for s in narrow:
                xg = compute_xg(
                    distance_m=s["distance_m"],
                    angle_deg=s["angle_deg"],
                    body_part=s["body_part"],
                    shot_type=s["shot_type"],
                )
                if xg > prev_xg + 0.05:
                    violations += 1
                prev_xg = xg
                total += 1
        assert violations / max(total, 1) < 0.25, f"Too many angle violations: {violations}/{total}"


class TestXgModelCount:
    """Ensure we have enough shots for statistically meaningful regression tests."""

    def test_sufficient_shots(self, all_shots):
        assert len(all_shots) >= 50, f"Only {len(all_shots)} shots loaded — need >= 50"
        assert len({s["match_id"] for s in all_shots}) >= 3, "Need >= 3 matches"
