"""Tests for FaceRecognitionService — InsightFace-based player identification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


def _normalized_emb(val: float = 0.5) -> np.ndarray:
    """Create a unit-normalized embedding for face matching tests."""
    raw = np.full(512, val, dtype=np.float32)
    norm = raw / max(np.linalg.norm(raw), 1e-8)
    return norm


def _install_insightface_stub():
    if "insightface" in sys.modules:
        return
    import types as t

    if_mod = t.ModuleType("insightface")
    app_mod = t.ModuleType("insightface.app")
    zoo_mod = t.ModuleType("insightface.model_zoo")

    class MockFace:
        def __init__(self):
            self.bbox = np.array([10, 10, 100, 130], dtype=np.float32)
            self.normed_embedding = np.ones(512, dtype=np.float32) * 0.01
            self.det_score = 0.95

    class MockFaceAnalysis:
        def __init__(self, *a, **k):
            pass

        def prepare(self, *a, **k):
            pass

        def get(self, img):
            return [MockFace()]

    app_mod.FaceAnalysis = MockFaceAnalysis
    zoo_mod.get_model = lambda *a, **k: MagicMock()
    if_mod.app = app_mod
    if_mod.model_zoo = zoo_mod
    sys.modules["insightface"] = if_mod
    sys.modules["insightface.app"] = app_mod
    sys.modules["insightface.model_zoo"] = zoo_mod


_install_insightface_stub()

_mod = load_service_module("face_rec_test", "face_recognition_service.py")
FaceRecognitionService = _mod.FaceRecognitionService


def _make_profile(pid: int, name: str, emb_list: list | None = None):
    p = {"id": pid, "display_name": name, "team": "home"}
    if emb_list is not None:
        p["face_embedding"] = json.dumps(emb_list)
        p["face_confidence"] = 0.9
    return p


def _make_gallery_embedding(val: float = 0.5):
    return np.full(512, val, dtype=np.float32)


# ===========================================================================
# Tests
# ===========================================================================


class TestAvailable:
    def test_available_when_insightface_installed(self):
        svc = FaceRecognitionService()
        assert svc.available is True

    def test_available_returns_false_when_not_installed(self):
        with patch.dict(sys.modules, {"insightface": None, "insightface.app": None}, clear=False):
            pass
        svc = FaceRecognitionService()
        with patch.dict("sys.modules"):
            sys.modules.pop("insightface", None)
            sys.modules.pop("insightface.app", None)
            assert svc.available is False


class TestDetectFaces:
    def test_detect_faces_returns_list(self):
        svc = FaceRecognitionService()
        svc._ensure_models()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = svc.detect_faces(img)
        assert isinstance(faces, list)
        for f in faces:
            assert "bbox" in f
            assert "embedding" in f
            assert "confidence" in f

    def test_detect_faces_returns_empty_when_app_none(self):
        svc = FaceRecognitionService()
        with patch.object(svc, "_ensure_models"):
            svc._app = None
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            assert svc.detect_faces(img) == []


class TestGetEmbedding:
    def test_get_embedding_returns_array(self):
        svc = FaceRecognitionService()
        svc._ensure_models()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        emb = svc.get_embedding(img)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (512,)

    def test_get_embedding_returns_none_when_no_faces(self):
        svc = FaceRecognitionService()
        with patch.object(svc, "_ensure_models"):
            svc._app = MagicMock()
            svc._app.get.return_value = []
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            assert svc.get_embedding(img) is None


class TestBuildGallery:
    def test_build_gallery_with_embeddings(self):
        svc = FaceRecognitionService()
        profiles = [
            _make_profile(1, "Alice", [0.1] * 512),
            _make_profile(2, "Bob", [0.2] * 512),
        ]
        svc.build_gallery(profiles)
        assert svc._gallery_loaded is True
        assert len(svc._gallery) == 2
        assert svc._gallery[0]["display_name"] == "Alice"

    def test_build_gallery_skips_profiles_without_embedding(self):
        svc = FaceRecognitionService()
        profiles = [
            _make_profile(1, "Alice"),
            _make_profile(2, "Bob", [0.1] * 512),
        ]
        svc.build_gallery(profiles)
        assert len(svc._gallery) == 1

    def test_build_gallery_skips_bad_json(self):
        svc = FaceRecognitionService()
        profiles = [{"id": 1, "display_name": "Bad", "face_embedding": "not-json"}]
        svc.build_gallery(profiles)
        assert len(svc._gallery) == 0

    def test_build_gallery_empty_profiles(self):
        svc = FaceRecognitionService()
        svc.build_gallery([])
        assert svc._gallery_loaded is True
        assert len(svc._gallery) == 0


class TestAddToGallery:
    def test_add_to_gallery(self):
        svc = FaceRecognitionService()
        emb = _make_gallery_embedding(0.3)
        svc.add_to_gallery(1, emb, 0.95)
        assert len(svc._gallery) == 1
        assert svc._gallery[0]["profile_id"] == 1
        assert svc._gallery_loaded is True


class TestMatchFace:
    def test_match_face_returns_best_match(self):
        svc = FaceRecognitionService()
        emb1 = _normalized_emb(0.5)
        emb2 = _normalized_emb(0.9)
        svc._gallery = [
            {
                "profile_id": 1,
                "display_name": "A",
                "jersey_number": 10,
                "team": "home",
                "embedding": emb1,
                "confidence": 0.9,
            },
            {
                "profile_id": 2,
                "display_name": "B",
                "jersey_number": 7,
                "team": "away",
                "embedding": emb2,
                "confidence": 0.8,
            },
        ]
        svc._gallery_loaded = True
        query = _normalized_emb(0.5)
        result = svc.match_face(query, threshold=0.12)
        assert result is not None
        assert result["profile_id"] == 1

    def test_match_face_returns_none_when_not_loaded(self):
        svc = FaceRecognitionService()
        svc._gallery_loaded = False
        result = svc.match_face(_normalized_emb(0.5))
        assert result is None

    def test_match_face_returns_none_when_empty_gallery(self):
        svc = FaceRecognitionService()
        svc._gallery_loaded = True
        svc._gallery = []
        result = svc.match_face(_normalized_emb(0.5))
        assert result is None

    def test_match_face_returns_none_when_exceeds_threshold(self):
        svc = FaceRecognitionService()
        svc._gallery = [
            {
                "profile_id": 1,
                "display_name": "A",
                "jersey_number": 10,
                "team": "home",
                "embedding": _normalized_emb(0.9),
                "confidence": 0.9,
            },
        ]
        svc._gallery_loaded = True
        query = _normalized_emb(0.0)
        result = svc.match_face(query, threshold=0.01)
        assert result is None

    def test_match_face_mismatched_shape_skips(self):
        svc = FaceRecognitionService()
        svc._gallery = [
            {
                "profile_id": 1,
                "display_name": "A",
                "jersey_number": 10,
                "team": "home",
                "embedding": np.ones(256, dtype=np.float32),
                "confidence": 0.9,
            },
        ]
        svc._gallery_loaded = True
        query = _normalized_emb(0.5)
        result = svc.match_face(query, threshold=0.5)
        assert result is None


class TestIdentifyPlayerFromCrop:
    def test_identify_player_from_crop_success(self):
        svc = FaceRecognitionService()
        svc._ensure_models()
        emb = _normalized_emb(0.5)
        svc._gallery = [
            {
                "profile_id": 1,
                "display_name": "Cristiano",
                "jersey_number": 7,
                "team": "home",
                "embedding": emb,
                "confidence": 0.9,
            },
        ]
        svc._gallery_loaded = True
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = svc.identify_player_from_crop(img)
        assert result is not None

    def test_identify_player_from_crop_no_face(self):
        svc = FaceRecognitionService()
        with patch.object(svc, "_ensure_models"):
            svc._app = MagicMock()
            svc._app.get.return_value = []
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = svc.identify_player_from_crop(img)
            assert result is None


class TestCropFaceRegion:
    def test_crop_face_region_returns_none_on_bad_bbox(self):
        svc = FaceRecognitionService()
        from types import SimpleNamespace

        frame_det = SimpleNamespace(image_width=100, image_height=100, frame_number=0)
        result = svc._crop_face_region(frame_det, [0, 0, 0, 0], None)
        assert result is None


class TestIdentifyPlayersInMatch:
    def test_identify_players_in_match_empty_gallery(self):
        svc = FaceRecognitionService()
        svc._ensure_models()
        profiles = []
        track_data = MagicMock()
        track_data.frames = []
        result = svc.identify_players_in_match(profiles, track_data)
        assert result == {}

    def test_identify_players_in_match_empty_frames(self):
        svc = FaceRecognitionService()
        svc._ensure_models()
        profiles = [_make_profile(1, "Test", [0.1] * 512)]
        track_data = MagicMock()
        track_data.frames = []
        result = svc.identify_players_in_match(profiles, track_data)
        assert result == {}
