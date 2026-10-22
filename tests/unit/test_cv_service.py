"""Tests for the CVService computer vision pipeline.

Covers: Detection dataclasses, pitch mask, dominant color, team clustering,
detect_frame filtering, process_video track filtering, swap_teams.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


# ---------------------------------------------------------------------------
# Module-level stubs for cv2 and sklearn (both absent from test env)
# ---------------------------------------------------------------------------


def _install_cv2_stub() -> None:
    if "cv2" in sys.modules:
        return
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLOR_BGR2HSV = 40
    cv2_stub.RETR_EXTERNAL = 0
    cv2_stub.CHAIN_APPROX_SIMPLE = 1
    cv2_stub.MORPH_CLOSE = 2
    cv2_stub.MORPH_OPEN = 3
    cv2_stub.CAP_PROP_FPS = 5
    cv2_stub.CAP_PROP_FRAME_COUNT = 7
    for fn in (
        "cvtColor",
        "inRange",
        "morphologyEx",
        "findContours",
        "contourArea",
        "drawContours",
        "calcHist",
        "GaussianBlur",
        "countNonZero",
        "VideoCapture",
        "setUseOptimized",
        "useOptimized",
    ):
        setattr(cv2_stub, fn, MagicMock())
    # Add cv2.ocl sub-module
    ocl_mod = types.ModuleType("cv2.ocl")
    ocl_mod.haveOpenCL = MagicMock(return_value=False)
    ocl_mod.setUseOpenCL = MagicMock()
    cv2_stub.ocl = ocl_mod
    sys.modules["cv2.ocl"] = ocl_mod
    sys.modules["cv2"] = cv2_stub


def _install_sklearn_stub() -> None:
    """Minimal sklearn stub so @patch('sklearn.cluster.KMeans') resolves."""
    if "sklearn" in sys.modules:
        return
    from importlib.machinery import ModuleSpec

    sk_mod = types.ModuleType("sklearn")
    sk_mod.__spec__ = ModuleSpec("sklearn", None)
    sk_cluster = types.ModuleType("sklearn.cluster")
    sk_cluster.__spec__ = ModuleSpec("sklearn.cluster", None)
    sk_cluster.KMeans = MagicMock()
    sk_mod.cluster = sk_cluster
    sys.modules["sklearn"] = sk_mod
    sys.modules["sklearn.cluster"] = sk_cluster


_install_cv2_stub()
_install_sklearn_stub()


@pytest.fixture(scope="module")
def cv_mod():
    """Load CVService module via conftest helper."""
    return load_service_module("kawkab.services.cv_service", "cv_service.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CpuTensor:
    """Minimal stand-in for an Ultralytics tensor: .cpu().numpy() -> ndarray."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _make_mock_boxes(xyxys, confs, cls_ids, track_ids=None):
    """Create a mock Ultralytics Boxes object."""
    boxes = MagicMock()
    boxes.__len__.return_value = len(xyxys)
    boxes.xyxy = [_CpuTensor(x) for x in xyxys]
    boxes.conf = [_CpuTensor(c) for c in confs]
    boxes.cls = [_CpuTensor(c) for c in cls_ids]
    if track_ids is not None:
        boxes.id = [_CpuTensor(t) for t in track_ids]
    else:
        boxes.id = None
    return boxes


def _make_track_schedule(schedule: dict[int, list[int]], bbox_map: dict[int, tuple] | None = None):
    """Build a detect_frame that returns specific track IDs per frame.

    Args:
        schedule: {track_id: [frame_numbers]}
        bbox_map: optional {track_id: (x1, y1, x2, y2)} for custom bboxes.
                  If omitted, each track gets a widely-spaced x-center
                  (cx = 10 + tid*60) to avoid false stitching by P0-A1.
    """

    def _factory(cv_mod):
        async def _detect(frame, frame_number, timestamp, norfair_tracker=None, period=1):
            dets = []
            for tid, frames in schedule.items():
                if frame_number in frames:
                    if bbox_map and tid in bbox_map:
                        bbox = bbox_map[tid]
                    else:
                        x_offset = 10 + tid * 60
                        bbox = (x_offset, 20, x_offset + 50, 120)
                    dets.append(
                        cv_mod.Detection(
                            bbox=bbox,
                            confidence=0.85,
                            class_id=0,
                            class_name="person",
                            track_id=tid,
                        )
                    )
            return cv_mod.FrameDetections(
                frame_number=frame_number,
                timestamp=timestamp,
                detections=dets,
                image_width=100,
                image_height=100,
            )

        return _detect

    return _factory


# ===================================================================
# MatchTrackData.swap_teams
# ===================================================================


class TestMatchTrackDataSwapTeams:
    """MatchTrackData.swap_teams swaps home/away assignments and metrics."""

    def test_swap_teams_basic(self, cv_mod):
        data = cv_mod.MatchTrackData(
            match_id=1,
            fps=30,
            total_frames=100,
            duration_seconds=10.0,
            frames=[],
            track_registry={},
            player_teams={1: "home", 2: "away", 3: "home"},
        )
        data.swap_teams()
        assert data.player_teams == {1: "away", 2: "home", 3: "away"}

    def test_swap_teams_with_metrics(self, cv_mod):
        data = cv_mod.MatchTrackData(
            match_id=1,
            fps=30,
            total_frames=100,
            duration_seconds=10.0,
            frames=[],
            track_registry={},
            player_teams={1: "home", 2: "away"},
            tracking_metrics={
                "team_detection": {
                    "home_avg_bgr": (50, 100, 200),
                    "away_avg_bgr": (200, 50, 100),
                    "home_size": 11,
                    "away_size": 11,
                },
            },
        )
        data.swap_teams()
        td = data.tracking_metrics["team_detection"]
        assert td["home_avg_bgr"] == (200, 50, 100)
        assert td["away_avg_bgr"] == (50, 100, 200)
        assert td["home_size"] == 11
        assert td["away_size"] == 11

    def test_swap_teams_partial_metrics(self, cv_mod):
        data = cv_mod.MatchTrackData(
            match_id=1,
            fps=30,
            total_frames=100,
            duration_seconds=10.0,
            frames=[],
            track_registry={},
            player_teams={1: "home", 2: "away"},
            tracking_metrics={"team_detection": {"home_size": 10}},
        )
        data.swap_teams()
        assert data.player_teams[1] == "away"
        assert data.player_teams[2] == "home"
        assert data.tracking_metrics["team_detection"]["home_size"] == 10

    def test_swap_teams_preserves_non_team_labels(self, cv_mod):
        data = cv_mod.MatchTrackData(
            match_id=1,
            fps=30,
            total_frames=100,
            duration_seconds=10.0,
            frames=[],
            track_registry={},
            player_teams={1: "referee", 2: "unknown"},
        )
        data.swap_teams()
        assert data.player_teams[1] == "referee"
        assert data.player_teams[2] == "unknown"


# ===================================================================
# _compute_pitch_mask
# ===================================================================


class TestComputePitchMask:
    """CVService._compute_pitch_mask returns a binary pitch mask or None."""

    def test_valid_frame_returns_mask(self, cv_mod):
        with (
            patch("cv2.cvtColor") as mock_cvt,
            patch("cv2.inRange") as mock_inrange,
            patch("cv2.morphologyEx") as mock_morph,
            patch("cv2.findContours") as mock_find,
            patch("cv2.contourArea") as mock_area,
            patch("cv2.countNonZero") as mock_count,
            patch("cv2.drawContours") as _,
        ):
            mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_inrange.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_morph.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_count.return_value = 5000
            cnt = np.array([[[0, 0]], [[0, 99]], [[99, 99]], [[99, 0]]], dtype=np.int32)
            mock_find.return_value = ([cnt], None)
            mock_area.return_value = 5000.0

            service = cv_mod.CVService(model_size="n")
            service._initialized = True
            service._pitch_hsv_range = (np.array([40, 40, 40]), np.array([80, 255, 255]))
            mask = service._compute_pitch_mask(np.zeros((100, 100, 3), dtype=np.uint8))
            assert mask is not None
            assert mask.dtype == np.bool_

    def test_no_contours_returns_none(self, cv_mod):
        with (
            patch("cv2.cvtColor") as mock_cvt,
            patch("cv2.inRange") as mock_inrange,
            patch("cv2.morphologyEx") as mock_morph,
            patch("cv2.findContours") as mock_find,
        ):
            mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_inrange.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_morph.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_find.return_value = ([], None)

            service = cv_mod.CVService(model_size="n")
            service._initialized = True
            assert service._compute_pitch_mask(np.zeros((100, 100, 3), dtype=np.uint8)) is None

    def test_exception_returns_none(self, cv_mod):
        with patch("cv2.cvtColor", side_effect=ValueError("mock error")):
            service = cv_mod.CVService(model_size="n")
            service._initialized = True
            assert service._compute_pitch_mask(np.zeros((100, 100, 3), dtype=np.uint8)) is None


# ===================================================================
# _get_dominant_color
# ===================================================================


class TestGetDominantColor:
    """CVService._get_dominant_color returns average BGR or None."""

    def test_valid_image_returns_bgr_tuple(self, cv_mod):
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        img = np.full((20, 20, 3), [100, 50, 200], dtype=np.uint8)
        assert service._get_dominant_color(img) == (100, 50, 200)

    def test_empty_image_returns_none(self, cv_mod):
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        assert service._get_dominant_color(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    def test_too_small_image_returns_none(self, cv_mod):
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        assert service._get_dominant_color(np.zeros((3, 3, 3), dtype=np.uint8)) is None

    def test_all_white_image_returns_none(self, cv_mod):
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        assert service._get_dominant_color(np.full((20, 20, 3), 255, dtype=np.uint8)) is None

    def test_all_black_image_returns_none(self, cv_mod):
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        assert service._get_dominant_color(np.zeros((20, 20, 3), dtype=np.uint8)) is None

    def test_few_colored_pixels_returns_none(self, cv_mod):
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        img = np.full((20, 20, 3), 255, dtype=np.uint8)
        img[0, 0] = [100, 100, 100]
        assert service._get_dominant_color(img) is None


# ===================================================================
# _cluster_team_colors
# ===================================================================


class TestClusterTeamColors:
    """CVService._cluster_team_colors labels tracks as home/away/referee."""

    def test_single_track_returns_default(self, cv_mod):
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        data = {1: {"primary_color": (100, 100, 100), "samples": 5}}
        assert service._cluster_team_colors(data, n_clusters=2) == {1: 0}

    @patch("sklearn.cluster.KMeans")
    def test_two_teams_with_sklearn(self, mock_kmeans, cv_mod):
        mock_km = MagicMock()
        mock_km.fit_predict.return_value = np.array([0, 1])
        mock_km.cluster_centers_ = np.array(
            [[200, 100, 50], [50, 100, 200]],
            dtype=np.float64,
        )
        mock_kmeans.return_value = mock_km

        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        data = {
            1: {"primary_color": (200, 100, 50), "samples": 5},
            2: {"primary_color": (50, 100, 200), "samples": 5},
        }
        result = service._cluster_team_colors(data, n_clusters=2)
        assert result[1] in ("home", "away")
        assert result[2] in ("home", "away")
        assert result[1] != result[2]

    @patch("sklearn.cluster.KMeans")
    def test_two_teams_home_is_brighter(self, mock_kmeans, cv_mod):
        mock_km = MagicMock()
        mock_km.fit_predict.return_value = np.array([0, 1])
        mock_km.cluster_centers_ = np.array(
            [[50, 100, 200], [200, 150, 50]],
            dtype=np.float64,
        )
        mock_kmeans.return_value = mock_km

        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        data = {
            1: {"primary_color": (50, 100, 200), "samples": 5},
            2: {"primary_color": (200, 150, 50), "samples": 5},
        }
        result = service._cluster_team_colors(data, n_clusters=2)
        assert result == {1: "away", 2: "home"}

    @patch("cv2.cvtColor")
    @patch("sklearn.cluster.KMeans")
    def test_three_clusters_with_referee(self, mock_kmeans, mock_cvt, cv_mod):
        mock_km = MagicMock()
        mock_km.fit_predict.return_value = np.array([0, 1, 2])
        mock_km.cluster_centers_ = np.array(
            [[80, 80, 80], [50, 100, 200], [200, 100, 50]],
            dtype=np.float64,
        )
        mock_kmeans.return_value = mock_km

        # Shape-preserving cvtColor stub: the LAB clustering path converts
        # BOTH the per-track colors (N,1,3) and the cluster centroids
        # (K,1,3); the old fixed-shape mock broke the centroid conversion.
        # Pass-through with a fixed BGR->HSV-style mapping per row keeps the
        # referee (low saturation) heuristic deterministic.
        def _cvt(img, code):
            arr = np.asarray(img, dtype=np.float32)
            out = arr.reshape(-1, 3).copy()
            # deterministic transform: keep first channel low (low "sat")
            out[:, 1] = out[:, 1] * 0.2
            return out.reshape(arr.shape)

        mock_cvt.side_effect = _cvt
        service = cv_mod.CVService(model_size="n")
        service._initialized = True
        data = {
            1: {"primary_color": (80, 80, 80), "samples": 5},
            2: {"primary_color": (50, 100, 200), "samples": 5},
            3: {"primary_color": (200, 100, 50), "samples": 5},
        }
        result = service._cluster_team_colors(data, n_clusters=3)
        assert result[1] == "referee"
        assert result[2] in ("home", "away")
        assert result[3] in ("home", "away")
        assert result[2] != result[3]

    def test_fallback_when_sklearn_unavailable(self, cv_mod):
        _real_import = builtins.__import__

        def _block_sklearn(name, *args, **kwargs):
            if name == "sklearn" or name.startswith("sklearn."):
                raise ImportError(f"No module named '{name}'")
            return _real_import(name, *args, **kwargs)

        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("sklearn")}
        try:
            with patch.object(builtins, "__import__", _block_sklearn):
                service = cv_mod.CVService(model_size="n")
                service._initialized = True
                data = {
                    1: {"primary_color": (50, 50, 200), "samples": 5},
                    2: {"primary_color": (200, 50, 50), "samples": 5},
                    3: {"primary_color": (100, 100, 100), "samples": 5},
                    4: {"primary_color": (180, 180, 50), "samples": 5},
                }
                result = service._cluster_team_colors(data, n_clusters=2)
        finally:
            sys.modules.update(saved)

        assert len(result) == 4
        assert set(result.values()) == {"home", "away"}
        home_count = sum(1 for v in result.values() if v == "home")
        away_count = sum(1 for v in result.values() if v == "away")
        assert home_count == away_count


# ===================================================================
# detect_frame
# ===================================================================


class TestDetectFrame:
    """CVService.detect_frame filtering and tracking."""

    def _build_service(self, cv_mod, **kwargs):
        params = {"model_size": "n", **kwargs}
        service = cv_mod.CVService(**params)
        service._initialized = True
        service._model = MagicMock()
        return service

    def _stub_results(self, service, boxes):
        service._model.track.return_value = [MagicMock(boxes=boxes)]

    @pytest.mark.asyncio
    async def test_basic_detection(self, cv_mod):
        service = self._build_service(cv_mod, confidence_threshold=0.3)
        service._model.names = {0: "person"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(10, 20, 40, 60)],
                [0.85],
                [0],
                track_ids=[1],
            ),
        )
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((200, 100), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((200, 100, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 1
        d = result.detections[0]
        assert d.class_name == "person"
        assert d.confidence == 0.85
        assert d.track_id == 1
        assert d.bbox == (10, 20, 40, 60)

    @pytest.mark.asyncio
    async def test_person_below_confidence_filtered(self, cv_mod):
        service = self._build_service(cv_mod, confidence_threshold=0.5)
        service._model.names = {0: "person"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(10, 20, 40, 60)],
                [0.3],
                [0],
            ),
        )
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((200, 100), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((200, 100, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_person_bbox_area_below_min_filtered(self, cv_mod):
        service = self._build_service(cv_mod, min_bbox_area_ratio=0.05)
        service._model.names = {0: "person"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(100, 100, 110, 105)],
                [0.8],
                [0],
            ),
        )
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((720, 540), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((720, 540, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_person_bbox_area_above_max_filtered(self, cv_mod):
        service = self._build_service(cv_mod, max_bbox_area_ratio=0.02)
        service._model.names = {0: "person"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(0, 0, 200, 100)],
                [0.8],
                [0],
            ),
        )
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((200, 100), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((200, 100, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_person_outside_pitch_mask_filtered(self, cv_mod):
        service = self._build_service(cv_mod)
        service._model.names = {0: "person"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(10, 20, 40, 60)],
                [0.85],
                [0],
            ),
        )
        mask = np.ones((200, 100), dtype=bool)
        mask[65, 25] = False  # foot = min(60+5, 199) = 65, x = int((10+40)/2) = 25
        with patch.object(service, "_compute_pitch_mask", return_value=mask):
            result = await service.detect_frame(
                np.zeros((200, 100, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_sports_ball_below_confidence_filtered(self, cv_mod):
        service = self._build_service(cv_mod, ball_confidence_threshold=0.5)
        service._model.names = {32: "sports ball"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(100, 100, 110, 110)],
                [0.3],
                [32],
            ),
        )
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((200, 200), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((200, 200, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_sports_ball_outside_pitch_mask_filtered(self, cv_mod):
        service = self._build_service(cv_mod)
        service._model.names = {32: "sports ball"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(100, 100, 110, 110)],
                [0.85],
                [32],
            ),
        )
        mask = np.ones((200, 200), dtype=bool)
        mask[105, 105] = False
        with patch.object(service, "_compute_pitch_mask", return_value=mask):
            result = await service.detect_frame(
                np.zeros((200, 200, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_multiple_detections_filtered_correctly(self, cv_mod):
        service = self._build_service(
            cv_mod, confidence_threshold=0.5, ball_confidence_threshold=0.2
        )
        service._model.names = {0: "person", 32: "sports ball"}
        self._stub_results(
            service,
            _make_mock_boxes(
                [(10, 20, 40, 60), (100, 100, 110, 110)],
                [0.85, 0.3],
                [0, 32],
            ),
        )
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((200, 200), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((200, 200, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 2

    @pytest.mark.asyncio
    async def test_empty_results_returns_no_detections(self, cv_mod):
        service = self._build_service(cv_mod)
        service._model.track.return_value = []
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((200, 200), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((200, 200, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_none_boxes_skips_parsing(self, cv_mod):
        service = self._build_service(cv_mod)
        mock_result = MagicMock()
        mock_result.boxes = None
        service._model.track.return_value = [mock_result]
        with patch.object(
            service, "_compute_pitch_mask", return_value=np.ones((200, 200), dtype=bool)
        ):
            result = await service.detect_frame(
                np.zeros((200, 200, 3), dtype=np.uint8),
                0,
                0.0,
            )
        assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_norfair_tracking_path(self, cv_mod):
        with (
            patch.object(cv_mod, "_NORFAIR_AVAILABLE", True),
            patch.object(cv_mod, "NorfairTracker"),
        ):
            service = self._build_service(cv_mod)
            service._model.names = {0: "person"}
            mock_boxes = _make_mock_boxes(
                [(10, 20, 40, 60)],
                [0.85],
                [0],
                track_ids=None,
            )
            service._model.return_value = [MagicMock(boxes=mock_boxes)]

            norfair_mock = MagicMock()
            norfair_mock.update.return_value = [
                {"track_id": 42, "bbox": (10, 20, 40, 60), "label": "person"},
            ]

            with patch.object(
                service, "_compute_pitch_mask", return_value=np.ones((200, 100), dtype=bool)
            ):
                result = await service.detect_frame(
                    np.zeros((200, 100, 3), dtype=np.uint8),
                    0,
                    0.0,
                    norfair_tracker=norfair_mock,
                )
            assert len(result.detections) == 1
            assert result.detections[0].track_id == 42


def _mock_video_capture(n_frames=90, fps=30.0, height=100, width=100):
    """Patch cv2.VideoCapture to yield n_frames then stop."""
    from unittest.mock import patch as _patch

    frames = [(True, np.zeros((height, width, 3), dtype=np.uint8)) for _ in range(n_frames)]
    frames.append((False, None))

    vc_patcher = _patch("cv2.VideoCapture")
    mock_vc = vc_patcher.start()
    mock_cap = MagicMock()
    mock_vc.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: fps if prop == 5 else n_frames
    mock_cap.read.side_effect = frames

    # Patch camera cut detector to prevent consuming the shared mock
    cc_patcher = _patch("kawkab.services.camera_cut_detector.CameraCutDetector.detect_cuts_fast")
    cc_mock = cc_patcher.start()
    cc_mock.return_value = []

    class _CombinedPatcher:
        def stop(self):
            cc_patcher.stop()
            vc_patcher.stop()

    return _CombinedPatcher()


# ===================================================================
# process_video
# ===================================================================


class TestProcessVideo:
    """CVService.process_video full pipeline and track filtering."""

    @pytest.mark.asyncio
    async def test_basic_processing(self, cv_mod):
        vc_patcher = _mock_video_capture(60)
        service = cv_mod.CVService(model_size="n", expected_player_count=22)
        service._initialized = True

        factory = _make_track_schedule({1: list(range(0, 60))})
        service.detect_frame = factory(cv_mod)

        result = await service.process_video(
            Path("/fake/video.mp4"),
            frame_skip=1,
            enable_team_detection=False,
        )
        vc_patcher.stop()

        assert result.fps == 30.0
        assert result.total_frames == 60
        assert len(result.frames) == 60
        assert 1 in result.track_registry
        assert result.track_registry[1]["frames_tracked"] == 60

    @pytest.mark.asyncio
    async def test_frame_skip_reuses_last_detections(self, cv_mod):
        vc_patcher = _mock_video_capture(30)
        service = cv_mod.CVService(
            model_size="n",
            expected_player_count=22,
            min_track_lifetime_frames=1,
        )
        service._initialized = True

        factory = _make_track_schedule({1: list(range(0, 30, 2))})
        service.detect_frame = factory(cv_mod)

        result = await service.process_video(
            Path("/fake/video.mp4"),
            frame_skip=2,
            enable_team_detection=False,
        )
        vc_patcher.stop()

        assert result.total_frames == 30
        assert result.track_registry[1]["frames_tracked"] == 15

    @pytest.mark.asyncio
    async def test_track_lifetime_filtering(self, cv_mod):
        vc_patcher = _mock_video_capture(20)
        service = cv_mod.CVService(
            model_size="n",
            expected_player_count=22,
            min_track_lifetime_frames=5,
        )
        service._initialized = True

        factory = _make_track_schedule(
            {
                1: list(range(0, 20)),
                2: [0, 5, 10],
            }
        )
        service.detect_frame = factory(cv_mod)

        result = await service.process_video(
            Path("/fake/video.mp4"),
            frame_skip=1,
            enable_team_detection=False,
        )
        vc_patcher.stop()

        assert 1 in result.track_registry
        assert 2 not in result.track_registry

    @pytest.mark.asyncio
    async def test_top_n_truncation(self, cv_mod):
        vc_patcher = _mock_video_capture(30)
        service = cv_mod.CVService(
            model_size="n",
            expected_player_count=10,
            min_track_lifetime_frames=1,
            max_keep_top_n=3,
        )
        service._initialized = True

        schedule: dict[int, list[int]] = {
            1: list(range(0, 30)),
            2: list(range(0, 28)),
            3: list(range(0, 25)),
            4: list(range(0, 20)),
            5: list(range(0, 10)),
        }
        factory = _make_track_schedule(schedule)
        service.detect_frame = factory(cv_mod)

        result = await service.process_video(
            Path("/fake/video.mp4"),
            frame_skip=1,
            enable_team_detection=False,
        )
        vc_patcher.stop()

        assert len(result.track_registry) == 3
        for tid in (1, 2, 3):
            assert tid in result.track_registry
        for tid in (4, 5):
            assert tid not in result.track_registry

    @pytest.mark.asyncio
    async def test_quality_assessment_excellent(self, cv_mod):
        """22 tracks with 100% lifetime → count_ratio=1.0 → excellent."""
        vc_patcher = _mock_video_capture(60)
        service = cv_mod.CVService(
            model_size="n",
            expected_player_count=22,
            min_track_lifetime_frames=1,
        )
        service._initialized = True

        factory = _make_track_schedule({i: list(range(0, 60)) for i in range(1, 23)})
        service.detect_frame = factory(cv_mod)

        result = await service.process_video(
            Path("/fake/video.mp4"),
            frame_skip=1,
            enable_team_detection=False,
        )
        vc_patcher.stop()

        assert result.tracking_metrics["tracking_quality"] == "excellent"

    @pytest.mark.asyncio
    async def test_match_type_inference_full_match(self, cv_mod):
        """360 frames @0.2fps = 1800s, between 1200–4800, low fragmentation
        + high avg_span → inferred as full_match."""
        vc_patcher = _mock_video_capture(n_frames=360, fps=0.2)
        service = cv_mod.CVService(
            model_size="n",
            expected_player_count=22,
            min_track_lifetime_frames=1,
        )
        service._initialized = True

        factory = _make_track_schedule({1: list(range(0, 360))})
        service.detect_frame = factory(cv_mod)

        result = await service.process_video(
            Path("/fake/video.mp4"),
            frame_skip=1,
            enable_team_detection=False,
        )
        vc_patcher.stop()

        # 360/0.2 = 1800s → between 1200 and 4800 → else branch
        # fragmentation = 1/1 = 1.0 (< 2.0) and avg_span ≈ 1795s (≥ 60)
        # → full_match
        assert result.match_type == "full_match"


# ===================================================================
# _detect_track_stitches
# ===================================================================


class TestDetectTrackStitches:
    """CVService._detect_track_stitches merges fragments of the same player."""

    def _make_service(self, cv_mod):
        svc = cv_mod.CVService(model_size="n")
        svc._initialized = True
        return svc

    def test_empty_valid_set(self, cv_mod):
        svc = self._make_service(cv_mod)
        result = svc._detect_track_stitches([], set(), {}, {}, 30.0, {})
        assert result == {}

    def test_no_stitch_needed(self, cv_mod):
        """Two tracks with disjoint time windows and far positions → no merge."""
        frames = [
            cv_mod.FrameDetections(
                0,
                0.0,
                [
                    cv_mod.Detection((0, 0, 20, 40), 0.9, 0, "person", track_id=1),
                ],
                100,
                100,
            ),
            cv_mod.FrameDetections(
                200,
                6.67,
                [
                    cv_mod.Detection((80, 0, 100, 40), 0.9, 0, "person", track_id=2),
                ],
                100,
                100,
            ),
        ]
        svc = self._make_service(cv_mod)
        result = svc._detect_track_stitches(
            frames,
            {1, 2},
            {1: 0, 2: 200},
            {1: 0, 2: 200},
            30.0,
            {},
            spatial_threshold_px=20.0,
            temporal_gap_max=1.0,
        )
        assert result == {}

    def test_overlap_stitch(self, cv_mod):
        """Two tracks overlapping in time with close x-centers → merge."""
        frames = []
        for fn in range(10):
            dets = [
                cv_mod.Detection((20, 20, 40, 60), 0.9, 0, "person", track_id=1),
                cv_mod.Detection((25, 20, 45, 60), 0.9, 0, "person", track_id=2),
            ]
            frames.append(cv_mod.FrameDetections(fn, fn / 30.0, dets, 100, 100))
        svc = self._make_service(cv_mod)
        result = svc._detect_track_stitches(
            frames,
            {1, 2},
            {1: 0, 2: 0},
            {1: 9, 2: 9},
            30.0,
            {},
            spatial_threshold_px=10.0,
            temporal_gap_max=1.0,
        )
        assert len(result) == 1
        discarded, survivor = list(result.items())[0]
        assert survivor == 1
        assert discarded == 2

    def test_temporal_gap_stitch(self, cv_mod):
        """Two sequential tracks with small gap and close boundary positions."""
        frames = []
        for fn in range(5):
            frames.append(
                cv_mod.FrameDetections(
                    fn,
                    fn / 30.0,
                    [
                        cv_mod.Detection((20, 20, 40, 60), 0.9, 0, "person", track_id=1),
                    ],
                    100,
                    100,
                )
            )
        for fn in range(8, 13):
            frames.append(
                cv_mod.FrameDetections(
                    fn,
                    fn / 30.0,
                    [
                        cv_mod.Detection((22, 20, 42, 60), 0.9, 0, "person", track_id=2),
                    ],
                    100,
                    100,
                )
            )
        svc = self._make_service(cv_mod)
        result = svc._detect_track_stitches(
            frames,
            {1, 2},
            {1: 0, 2: 8},
            {1: 4, 2: 12},
            30.0,
            {},
            spatial_threshold_px=10.0,
            temporal_gap_max=5.0,
        )
        assert len(result) == 1
        discarded, survivor = list(result.items())[0]
        assert survivor == 1

    def test_transitive_resolution(self, cv_mod):
        """Three overlapping tracks → two get merged (no transitive chains)."""
        frames = []
        for fn in range(10):
            dets = [
                cv_mod.Detection((20, 20, 40, 60), 0.9, 0, "person", track_id=1),
                cv_mod.Detection((25, 20, 45, 60), 0.9, 0, "person", track_id=2),
                cv_mod.Detection((28, 20, 48, 60), 0.9, 0, "person", track_id=3),
            ]
            frames.append(cv_mod.FrameDetections(fn, fn / 30.0, dets, 100, 100))
        svc = self._make_service(cv_mod)
        result = svc._detect_track_stitches(
            frames,
            {1, 2, 3},
            {1: 0, 2: 0, 3: 0},
            {1: 9, 2: 9, 3: 9},
            30.0,
            {},
            spatial_threshold_px=10.0,
            temporal_gap_max=1.0,
        )
        assert len(result) >= 1
        for _discarded, survivor in result.items():
            assert survivor not in result  # no transitive chains remain

    @pytest.mark.asyncio
    async def test_stitch_in_process_video(self, cv_mod):
        """Integration: overlapping tracks in process_video get stitched."""
        vc_patcher = _mock_video_capture(30)
        service = cv_mod.CVService(
            model_size="n",
            expected_player_count=22,
            min_track_lifetime_frames=1,
        )
        service._initialized = True

        # Two tracks with overlapping frames and close x-positions
        schedule: dict[int, list[int]] = {
            1: list(range(0, 30)),
            2: list(range(5, 25)),
        }
        bbox_map = {1: (20, 20, 40, 60), 2: (25, 20, 45, 60)}
        factory = _make_track_schedule(schedule, bbox_map=bbox_map)
        service.detect_frame = factory(cv_mod)

        result = await service.process_video(
            Path("/fake/video.mp4"),
            frame_skip=1,
            enable_team_detection=False,
        )
        vc_patcher.stop()

        # Track 2 should be stitched into track 1
        assert result.tracking_metrics.get("stitched_tracks", 0) >= 1
        # Only one final track
        assert len(result.track_registry) == 1
        assert 1 in result.track_registry
