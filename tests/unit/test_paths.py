"""Tests for paths module."""

from __future__ import annotations

from kawkab.core.paths import Paths, get_paths


def test_paths_creation(tmp_path, monkeypatch) -> None:
    """Test that all required directories are created."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "userprofile"))

    paths = Paths()

    assert paths.appdata.exists()
    assert paths.localappdata.exists()
    assert paths.documents.exists()
    assert paths.videos.exists()
    assert paths.exports.exists()
    assert paths.cache.exists()
    assert paths.logs.exists()
    assert paths.models.exists()


def test_paths_singleton() -> None:
    """Test get_paths returns same instance."""
    p1 = get_paths()
    p2 = get_paths()
    assert p1 is p2


def test_database_path() -> None:
    """Test database path is a file under appdata."""
    paths = get_paths()
    assert paths.database.parent == paths.appdata
    assert paths.database.suffix == ".db"


def test_knowledge_base_path_exists() -> None:
    """Test knowledge base path exists in package."""
    paths = get_paths()
    kb_path = paths.knowledge_base
    assert kb_path.exists()
    assert kb_path.is_dir()
