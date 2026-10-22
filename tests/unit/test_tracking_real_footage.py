"""Regression tests for real-footage tracking fixes (2026-09 validation round).

These pin three behaviors found broken ONLY on real footage
(data/real_match.mp4) -- synthetic tests passed while the real pipeline
produced 1 usable track out of a 22-player match:

1. Boxmot ID mapping must use boxmot's own ``det_ind`` column, with a
   per-class IoU fallback -- NOT a global geometric re-match (Kalman drift
   during pans made IoU < threshold and dropped IDs on every fast pan).
   Also: the tracker must be updated even on zero-detection frames.
2. The pitch mask must be self-healing: a one-time HSV calibration on a
   blank/intro frame used to poison every later frame (7% garbage mask
   passed the old 5% sanity check; the pitch gate then discarded 100% of
   player detections).
3. Team clustering must use pooled per-sample clustering + majority vote.
   Per-track means collapsed under occlusion/stitch contamination into a
   single cluster, leaving zero team assignments on real kits.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import install_kawkab_stubs

install_kawkab_stubs()

import sys

# test_cv_service installs a minimal sklearn stub at collection time on the
# stale premise that sklearn is absent from the env (it is present -- the
# pooled clustering under test here needs the REAL K-means). Restore it.
for _m in ("sklearn.cluster", "sklearn"):
    sys.modules.pop(_m, None)
import sklearn.cluster  # noqa: E402,F401  # noqa: E402

from tests.conftest import load_service_module  # noqa: E402


@pytest.fixture(scope="module")
def cv_mod():
    return load_service_module("kawkab.services.cv_service", "cv_service.py")


def _make_service(cv_mod):
    svc = cv_mod.CVService(model_size="n")
    svc._initialized = True
    return svc


# ---------------------------------------------------------------------------
# 1. boxmot det_ind mapping
# ---------------------------------------------------------------------------


def _boxmot_row(x1, y1, x2, y2, tid, conf, cls, det_ind):
    return [x1, y1, x2, y2, tid, conf, cls, det_ind]


class _FakeTracker:
    """Returns a fixed boxmot-style output regardless of input."""

    def __init__(self, rows):
        self._rows = np.array(rows, dtype=np.float32)
        self.update_calls: list[np.ndarray] = []

    def update(self, dets, frame):
        self.update_calls.append(np.asarray(dets))
        return self._rows


class _FakeBoxes:
    """Minimal Ultralytics Boxes stand-in (cpu().numpy() support)."""

    def __init__(self, xyxys, confs, cls_ids):
        arr = np.hstack(
            [
                np.array(xyxys, dtype=np.float32),
                np.array(confs, dtype=np.float32)[:, None],
                np.array(cls_ids, dtype=np.float32)[:, None],
            ]
        )
        self._arr = arr
        self.id = None

    @property
    def xyxy(self):
        return _CpuTensor(self._arr[:, :4])

    @property
    def conf(self):
        return _CpuTensor(self._arr[:, 4])

    @property
    def cls(self):
        return _CpuTensor(self._arr[:, 5])

    def __len__(self):
        return len(self._arr)


class _CpuTensor:
    """Stand-in for an Ultralytics tensor: supports row indexing and
    .cpu().numpy() -- the service slices rows (boxes.xyxy[i]) before
    converting."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def __getitem__(self, idx):
        return _CpuTensor(self._arr[idx])

    def __len__(self):
        return len(self._arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _patch_model(cv_mod, svc, boxes):
    """Mock the YOLO model so detect_frame runs its boxmot branch."""
    names = {0: "person", 32: "sports ball"}
    inner = MagicMock(boxes=boxes)
    inner.__len__.return_value = 1
    results = [inner]
    svc._model = MagicMock(return_value=results)
    svc._model.names = names
    return results


class TestBoxmotDetIndMapping:
    """boxmot output rows carry det_ind -- IDs must map directly."""

    def test_det_ind_maps_ids_without_geomatch(self, cv_mod):
        svc = _make_service(cv_mod)
        svc._boxmot_tracker = _FakeTracker([_boxmot_row(10, 10, 50, 60, 7, 0.9, 0, 1)])
        # Detection 1 is far from the track box (Kalman drift) -- the old
        # Hungarian re-match (IoU gate 0.3) dropped this ID entirely.
        boxes = _FakeBoxes([[100, 100, 140, 150], [12, 12, 52, 62]], [0.9, 0.9], [0, 0])
        _patch_model(cv_mod, svc, boxes)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with patch.object(cv_mod.CVService, "_compute_pitch_mask", return_value=None):
            dets = _run_detect(cv_mod, svc, frame)
        ids = sorted(d.track_id for d in dets if d.track_id is not None)
        assert 7 in ids, "det_ind mapping must survive Kalman box drift"

    def test_minority_class_falls_back_to_iou(self, cv_mod):
        svc = _make_service(cv_mod)
        # Ball row: det_ind=5 is stale/wrong (out of range for 2 dets and
        # class mismatch) -> per-class IoU fallback should assign det 0.
        svc._boxmot_tracker = _FakeTracker([_boxmot_row(200, 200, 210, 210, 3, 0.8, 32, 5)])
        boxes = _FakeBoxes([[200, 200, 210, 210], [12, 12, 52, 62]], [0.9, 0.9], [32, 0])
        _patch_model(cv_mod, svc, boxes)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with patch.object(cv_mod.CVService, "_compute_pitch_mask", return_value=None):
            dets = _run_detect(cv_mod, svc, frame)
        ball_ids = [d.track_id for d in dets if d.class_name == "sports ball"]
        assert ball_ids == [3], "stale det_ind must fall back to per-class IoU"

    def test_empty_frame_still_updates_tracker(self, cv_mod):
        """Zero-detection frames must still call tracker.update (boxmot
        requires one call per frame; skipping stalls its internal clock)."""
        svc = _make_service(cv_mod)
        tracker = _FakeTracker(np.empty((0, 8), dtype=np.float32))
        svc._boxmot_tracker = tracker
        svc._model = MagicMock(return_value=[])
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with patch.object(cv_mod.CVService, "_compute_pitch_mask", return_value=None):
            _run_detect(cv_mod, svc, frame)
        assert len(tracker.update_calls) == 1
        assert tracker.update_calls[0].shape == (0, 6)


def _run_detect(cv_mod, svc, frame):
    import asyncio

    return asyncio.run(svc.detect_frame(frame, 0, 0.0, period=1)).detections


# ---------------------------------------------------------------------------
# 2. Self-healing pitch mask
# ---------------------------------------------------------------------------


class TestSelfHealingPitchMask:
    """A garbage first-frame calibration must not poison later frames."""

    def _frame_with_grass(self, cv_mod, h=360, w=640):
        # Real-grass-like HSV: H~48, S~115, V~87 (matches measured footage).
        hsv = np.full((h, w, 3), (48, 115, 87), dtype=np.uint8)
        real_cv2 = pytest.importorskip("cv2")
        return real_cv2.cvtColor(hsv, real_cv2.COLOR_HSV2BGR)

    def test_blank_first_frame_does_not_poison(self, cv_mod):
        real_cv2 = pytest.importorskip("cv2")
        svc = _make_service(cv_mod)
        blank = np.zeros((360, 640, 3), dtype=np.uint8)  # the intro frame
        grass = self._frame_with_grass(cv_mod)

        mask_bad = svc._compute_pitch_mask(blank)
        # Blank frame: honest None (gate disabled), never a tiny garbage mask.
        if mask_bad is not None:
            assert mask_bad.mean() >= 0.05, "garbage mask must not pass sanity"

        mask_good = svc._compute_pitch_mask(grass)
        assert mask_good is not None
        assert mask_good.mean() > 0.5, "real pitch frame must yield a large mask"
        # And the cache must now be calibrated on grass, not the blank frame.
        rng = svc._pitch_hsv_range
        assert rng is not None
        probe = real_cv2.inRange(real_cv2.cvtColor(grass, real_cv2.COLOR_BGR2HSV), rng[0], rng[1])
        assert probe.mean() / 255.0 >= 0.10

    def test_tiny_mask_returns_none_not_garbage(self, cv_mod):
        svc = _make_service(cv_mod)
        svc._pitch_hsv_range = None
        # Uniform color that matches neither the calibrated nor hardcoded
        # range -> no trustworthy pitch -> None (gate disabled).
        frame = np.full((360, 640, 3), 128, dtype=np.uint8)
        assert svc._compute_pitch_mask(frame) is None


# ---------------------------------------------------------------------------
# 3. Pooled per-sample team clustering
# ---------------------------------------------------------------------------


class TestPooledTeamClustering:
    """Per-track means collapse; pooled samples + majority vote must not."""

    def _real_cv2(self):
        return pytest.importorskip("cv2")

    def _hsv_to_bgr(self, cv2, h, s, v):
        return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])

    def test_minority_kit_survives_pooling(self, cv_mod):
        """29 blue samples among 183 red must win their tracks' majority --
        the LAB per-track-mean path assigned ALL tracks to one cluster."""
        cv2 = self._real_cv2()
        svc = _make_service(cv_mod)
        red = self._hsv_to_bgr(cv2, 178, 150, 120)
        blue = self._hsv_to_bgr(cv2, 110, 150, 120)

        def noisy(color, n, seed):
            rng = np.random.default_rng(seed)
            return [
                tuple(int(np.clip(c + rng.integers(-12, 13), 0, 255)) for c in color)
                for _ in range(n)
            ]

        color_data = {}
        for i in range(10):  # red team
            color_data[100 + i] = {
                "primary_color": red,
                "samples": 18,
                "samples_list": noisy(red, 18, seed=i),
            }
        for i in range(2):  # blue team (the minority that kept vanishing)
            color_data[200 + i] = {
                "primary_color": red,  # per-track mean IS red (contaminated)
                "samples": 15,
                "samples_list": noisy(blue, 14, seed=50 + i) + noisy(red, 1, seed=99),
            }
        result = svc._cluster_team_colors(color_data, n_clusters=2)
        teams = sorted({result.get(t) for t in (200, 201)} - {None})
        assert teams == ["home"] or teams == ["away"], (
            "minority-kit tracks must land on one shared team label"
        )
        red_teams = {result.get(t) for t in range(100, 110)} - {None}
        assert red_teams and red_teams.isdisjoint(set(teams)), (
            "majority kit must not share the minority kit's label"
        )

    def test_tie_votes_excluded(self, cv_mod):
        cv2 = self._real_cv2()
        svc = _make_service(cv_mod)
        red = self._hsv_to_bgr(cv2, 178, 150, 120)
        blue = self._hsv_to_bgr(cv2, 110, 150, 120)
        color_data = {
            1: {
                "primary_color": red,
                "samples": 2,
                "samples_list": [red, blue],  # 1-1 tie
            },
            2: {"primary_color": red, "samples": 4, "samples_list": [red] * 4},
            3: {"primary_color": blue, "samples": 4, "samples_list": [blue] * 4},
        }
        result = svc._cluster_team_colors(color_data, n_clusters=2)
        assert 1 not in result, "genuinely tied track must not be guessed"

    def test_without_samples_list_legacy_path_intact(self, cv_mod):
        """Entries without raw samples use the per-track-mean path (old
        behavior and existing tests keep working)."""
        from unittest.mock import MagicMock

        mock_km = MagicMock()
        mock_km.fit_predict.return_value = np.array([0, 1])
        mock_km.cluster_centers_ = np.array([[200, 100, 50], [50, 100, 200]], dtype=np.float64)
        with patch("sklearn.cluster.KMeans", return_value=mock_km):
            data = {
                1: {"primary_color": (200, 100, 50), "samples": 5},
                2: {"primary_color": (50, 100, 200), "samples": 5},
            }
            svc = _make_service(cv_mod)
            result = svc._cluster_team_colors(data, n_clusters=2)
        assert set(result.values()) <= {"home", "away"}
        assert result[1] != result[2]
