"""Tests for ModelManager - model download, caching, and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from kawkab.core.model_manager import ModelManager


class TestModelManager:
    """Test model cache management."""

    def test_init_creates_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir) / "models")
            assert mgr.cache_dir.exists()

    def test_get_model_path_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            assert mgr.get_model_path("yolo11l") is None

    def test_is_model_available_false_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            assert mgr.is_model_available("yolo11l") is False

    def test_is_model_available_true_when_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            fake_model = Path(tmpdir) / "yolo11l.pt"
            fake_model.write_bytes(b"fake")
            assert mgr.is_model_available("yolo11l") is True

    def test_list_cached_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            (Path(tmpdir) / "yolo11n.pt").write_bytes(b"a")
            (Path(tmpdir) / "yolo11l.pt").write_bytes(b"b")
            models = mgr.list_cached_models()
            assert sorted(models) == ["yolo11l", "yolo11n"]

    def test_cleanup_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            (Path(tmpdir) / "yolo11n.pt").write_bytes(b"a")
            (Path(tmpdir) / "yolo11l.pt").write_bytes(b"b")
            removed = mgr.cleanup_cache(keep_models=["yolo11l"])
            assert removed == 1
            assert not (Path(tmpdir) / "yolo11n.pt").exists()
            assert (Path(tmpdir) / "yolo11l.pt").exists()

    def test_get_cache_size_mb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            (Path(tmpdir) / "yolo11l.pt").write_bytes(b"x" * 1024 * 1024)  # 1 MB
            assert mgr.get_cache_size_mb() == pytest.approx(1.0, rel=0.01)

    def test_ensure_model_returns_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            fake_model = Path(tmpdir) / "yolo11l.pt"
            fake_model.write_bytes(b"fake")
            path = mgr.ensure_model("yolo11l")
            assert path == fake_model

    def test_manifest_saved_after_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            # Fake a download by writing file directly
            fake_model = Path(tmpdir) / "yolo11n.pt"
            fake_model.write_bytes(b"fake_model_data")
            mgr._manifest["yolo11n"] = {
                "path": str(fake_model),
                "size_bytes": len(b"fake_model_data"),
            }
            mgr._save_manifest()
            assert mgr.manifest_path.exists()

    def test_compute_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"hello")
            sha = mgr._compute_sha256(test_file)
            assert len(sha) == 64  # SHA-256 hex length
            assert sha == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


class TestManifestTamperEvidence:
    """The models.json manifest is HMAC-signed; a hand-edited manifest must
    never be able to bless a swapped model binary (Phase 3.1)."""

    def test_saved_manifest_is_signed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            mgr._manifest["yolo11n"] = {"path": "x", "size_bytes": 1, "sha256": "a" * 64}
            mgr._save_manifest()
            data = json.loads(mgr.manifest_path.read_text())
            assert "models" in data and "hmac" in data
            assert len(data["hmac"]) == 64

    def test_tampered_manifest_is_discarded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            mgr._manifest["yolo11n"] = {"path": "x", "size_bytes": 1, "sha256": "a" * 64}
            mgr._save_manifest()

            # Attacker edits the manifest after signing
            data = json.loads(mgr.manifest_path.read_text())
            data["models"]["yolo11n"]["sha256"] = "b" * 64
            mgr.manifest_path.write_text(json.dumps(data))

            mgr2 = ModelManager(cache_dir=Path(tmpdir))
            assert mgr2._manifest == {}

    def test_untampered_manifest_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(cache_dir=Path(tmpdir))
            mgr._manifest["yolo11n"] = {"path": "x", "size_bytes": 1, "sha256": "a" * 64}
            mgr._save_manifest()
            mgr2 = ModelManager(cache_dir=Path(tmpdir))
            assert "yolo11n" in mgr2._manifest

    def test_every_default_model_has_pinned_checksum(self):
        for name, info in ModelManager.DEFAULT_MODELS.items():
            assert info["sha256"], f"{name} has no pinned sha256"
            assert len(info["sha256"]) == 64
            assert info["url"].startswith("https://")
