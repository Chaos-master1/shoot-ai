"""Tests for ClipLibraryService (v0.8.2+).

Tests clip extraction, playlist creation, and storage integration.

.. note::
    Renamed from ``test_clip_extraction_service`` import: the class
    ``ClipExtractionService`` in ``kawkab.services.clip_extraction_service``
    was renamed to ``ClipLibraryService`` in cycle F of ITERATION_LOG.md to
    disambiguate from the production ``clip_service.ClipExtractionService``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.clip_extraction_service import (
    ClipLibraryService,
    VideoClip,
)
from kawkab.services.storage_service import StorageService


class TestClipLibraryService:
    """Test video clip extraction and library management functionality."""

    def test_init_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "clips"
            _ = ClipLibraryService(output_dir=output_dir)
            assert output_dir.exists()

    def test_get_clip_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = ClipLibraryService(output_dir=Path(tmpdir))
            path = svc._get_clip_path(1, 5, "goal")
            assert path.name == "goal_5.mp4"
            assert path.parent.name == "match_1"

    def test_get_thumbnail_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = ClipLibraryService(output_dir=Path(tmpdir))
            path = svc._get_thumbnail_path(1, 5)
            assert path.name == "thumb_5.jpg"

    def test_extract_clip_with_invalid_duration(self):
        svc = ClipLibraryService()
        result = svc.extract_clip("/fake.mp4", 10.0, 5.0, "/out.mp4")
        assert result is False

    @pytest.mark.asyncio
    async def test_create_clip_from_event_missing_video(self):
        svc = ClipLibraryService()
        event = {"type": "goal", "timestamp": 60.0, "video_path": "/nonexistent.mp4"}
        result = await svc.create_clip_from_event(1, event)
        assert result is None

    @pytest.mark.asyncio
    async def test_create_clips_from_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = ClipLibraryService(output_dir=Path(tmpdir))
            events = [
                {"type": "goal", "timestamp": 60.0, "video_path": "/nonexistent1.mp4"},
                {"type": "shot", "timestamp": 120.0, "video_path": "/nonexistent2.mp4"},
            ]
            clips = await svc.create_clips_from_events(1, events)
            # All fail because videos don't exist, but method runs
            assert clips == []

    @pytest.mark.asyncio
    async def test_create_playlist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            svc = ClipLibraryService(storage_service=storage)
            playlist = await svc.create_playlist(
                "Best Goals", [1, 2, 3], "Top goals from the match"
            )
            assert playlist is not None
            assert playlist.name == "Best Goals"
            assert playlist.clip_ids == [1, 2, 3]
            assert playlist.playlist_id > 0

            await storage.close()

    @pytest.mark.asyncio
    async def test_save_clip_to_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = StorageService()
            storage._db_path = db_path
            await storage.initialize()

            clip_id = await storage.save_clip(
                {
                    "match_id": 1,
                    "event_type": "goal",
                    "start_seconds": 55.0,
                    "end_seconds": 65.0,
                    "duration_seconds": 10.0,
                    "source_video_path": "/match.mp4",
                    "output_path": "/clips/goal_1.mp4",
                    "thumbnail_path": "/clips/thumb_1.jpg",
                    "player_id": 7,
                    "description": "Amazing goal",
                }
            )
            assert clip_id > 0

            clips = await storage.get_clips_for_match(1)
            assert len(clips) == 1
            assert clips[0]["event_type"] == "goal"
            assert clips[0]["player_id"] == 7

            await storage.close()

    def test_dataclass_serialization(self):
        clip = VideoClip(
            clip_id=1,
            match_id=1,
            event_type="goal",
            start_seconds=55.0,
            end_seconds=65.0,
            duration_seconds=10.0,
            source_video_path="/match.mp4",
            output_path="/clips/goal_1.mp4",
        )
        d = clip.to_dict()
        assert d["event_type"] == "goal"
        assert d["duration_seconds"] == 10.0
