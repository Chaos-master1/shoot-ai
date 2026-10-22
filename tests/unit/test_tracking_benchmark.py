"""Tests for the production tracking benchmark.

Covers the benchmark's contract pieces so the published MOTA/IDF1 numbers
stay trustworthy:
  - GT loading (deterministic ids, frame limits, NaN handling)
  - detection degrader determinism (same seed → identical detections)
  - drop / false-positive parameter effects
  - end-to-end smoke with the REAL production tracker + real norfair:
    the ceiling row must strictly beat the degraded row on MOTA and IDF1.

Isolation note: test_norfair_tracker.py installs a norfair *stub* into
sys.modules at import time (documented in CLAUDE.md as a fragile pattern).
The real-tracker smoke test here therefore pops any stub before importing
kawkab.services.norfair_tracker and RESTORES the previous sys.modules state
afterwards — so this file neither depends on import order nor leaks real
norfair into other tests (the leak direction that bit test_cv_service).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_production_tracking.py"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "tracking"

_MODULE_NAME = "_benchmark_production_tracking"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load_benchmark_module()


@pytest.fixture()
def real_norfair_env():
    """Pop norfair stubs/cached modules so the REAL package loads, then restore."""
    keys = (
        "norfair",
        "norfair.camera_motion",
        "norfair.tracker",
        "kawkab.services.norfair_tracker",
    )
    saved = {k: sys.modules.pop(k) for k in keys if k in sys.modules}
    yield
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules.update(saved)


# ── GT loading ──────────────────────────────────────────────────────────────


class TestLoadGtTracks:
    def test_loads_players_with_deterministic_ids(self, bench):
        if not FIXTURE_DIR.exists():
            pytest.skip("tracking fixtures not present")
        gt = bench.load_gt_tracks(FIXTURE_DIR, frame_limit=200, px_per_m=20.0)
        assert len(gt) > 10  # both teams present
        home = [t for t in gt if 1000 <= t < 2000]
        away = [t for t in gt if t >= 2000]
        assert home and away
        # Deterministic across calls (sorted labels, never hash()).
        gt2 = bench.load_gt_tracks(FIXTURE_DIR, frame_limit=200, px_per_m=20.0)
        assert gt == gt2

    def test_frame_limit_is_respected(self, bench):
        if not FIXTURE_DIR.exists():
            pytest.skip("tracking fixtures not present")
        gt = bench.load_gt_tracks(FIXTURE_DIR, frame_limit=50, px_per_m=20.0)
        assert gt
        assert max(f for positions in gt.values() for f, _, _ in positions) <= 50

    def test_no_nan_positions(self, bench):
        if not FIXTURE_DIR.exists():
            pytest.skip("tracking fixtures not present")
        gt = bench.load_gt_tracks(FIXTURE_DIR, frame_limit=100, px_per_m=20.0)
        for positions in gt.values():
            for _, x, y in positions:
                assert x == x and y == y


# ── Degrader ────────────────────────────────────────────────────────────────


class TestBuildDetections:
    @pytest.fixture()
    def small_gt(self, bench):
        if not FIXTURE_DIR.exists():
            pytest.skip("tracking fixtures not present")
        return bench.load_gt_tracks(FIXTURE_DIR, frame_limit=60, px_per_m=20.0)

    def test_same_seed_is_byte_identical(self, bench, small_gt):
        a = bench.build_detections(
            small_gt, seed=13, noise_px_std=3.0, drop_rate=0.1, fp_rate=0.03, max_frame=60
        )
        b = bench.build_detections(
            small_gt, seed=13, noise_px_std=3.0, drop_rate=0.1, fp_rate=0.03, max_frame=60
        )
        assert a == b

    def test_different_seed_differs(self, bench, small_gt):
        a = bench.build_detections(
            small_gt, seed=13, noise_px_std=3.0, drop_rate=0.1, fp_rate=0.03, max_frame=60
        )
        b = bench.build_detections(
            small_gt, seed=99, noise_px_std=3.0, drop_rate=0.1, fp_rate=0.03, max_frame=60
        )
        assert a != b

    def test_total_drop_rate_leaves_only_false_positives(self, bench, small_gt):
        dets = bench.build_detections(
            small_gt, seed=13, noise_px_std=0.0, drop_rate=1.0, fp_rate=0.0, max_frame=60
        )
        assert dets == {}

    def test_false_positive_injection(self, bench, small_gt):
        n_players = len(small_gt)
        dets = bench.build_detections(
            small_gt, seed=13, noise_px_std=0.0, drop_rate=1.0, fp_rate=0.5, max_frame=60
        )
        # ~0.5 * n_players FPs per frame, every frame 1..60 present.
        assert len(dets) == 60
        for _frame, boxes in dets.items():
            assert 0 < len(boxes) <= 2 * n_players

    def test_ceiling_detections_are_exact_boxes(self, bench, small_gt):
        dets = bench.build_detections(
            small_gt, seed=13, noise_px_std=0.0, drop_rate=0.0, fp_rate=0.0, max_frame=60
        )
        total = sum(len(v) for v in dets.values())
        assert total == sum(len(p) for p in small_gt.values())


# ── End-to-end with the REAL production tracker ─────────────────────────────


class TestProductionTrackerEndToEnd:
    def test_ceiling_beats_degraded(self, bench, real_norfair_env):
        if not FIXTURE_DIR.exists():
            pytest.skip("tracking fixtures not present")
        pytest.importorskip("norfair", reason="real norfair not installed")
        # 120 frames: the tracker's 3-frame init costs a fixed ~66 FN, which
        # is 3.75% of an 80-frame window (ceiling MOTA 0.899) but only 2.5%
        # at 120 (0.933). Long enough that init overhead isn't the signal.
        gt = bench.load_gt_tracks(FIXTURE_DIR, frame_limit=120, px_per_m=20.0)

        ceiling = bench.build_detections(
            gt, seed=13, noise_px_std=0.0, drop_rate=0.0, fp_rate=0.0, max_frame=120
        )
        degraded = bench.build_detections(
            gt, seed=13, noise_px_std=3.0, drop_rate=0.10, fp_rate=0.02, max_frame=120
        )

        row_c = bench.run_tracker_row("ceiling", ceiling, gt, max_frame=120, match_threshold=40.0)
        row_d = bench.run_tracker_row("degraded", degraded, gt, max_frame=120, match_threshold=40.0)

        # The benchmark's core honesty guard: perfect detections must score
        # strictly better than noisy ones on both accuracy and identity.
        assert row_c["mota"] > row_d["mota"]
        assert row_c["idf1"] > row_d["idf1"]
        # Ceiling with zero detector error should be near-perfect on this
        # deterministic subset (verified: 0.99+); allow generous margin.
        assert row_c["mota"] > 0.9
        assert row_c["idf1"] > 0.9
        # Degraded must not collapse (this catches unphysical degradation
        # configs — the 10px-jitter bug that zeroed the row initially).
        assert row_d["mota"] > 0.5
        # Metric keys the MODEL_CARD publishes must all be present.
        for key in (
            "mota",
            "motp",
            "idf1",
            "id_switches",
            "fragments",
            "false_positives",
            "false_negatives",
        ):
            assert key in row_c and key in row_d
