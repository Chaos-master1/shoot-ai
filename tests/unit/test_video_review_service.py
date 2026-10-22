"""Tests for video review service."""

from __future__ import annotations

from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.video_review_service import (  # noqa: E402
    Annotation,
    AnnotationKind,
    Clip,
    ReviewSession,
    VideoReviewService,
)


class TestCreateSession:
    def test_creates_session_with_default_fps(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        assert isinstance(session, ReviewSession)
        assert session.match_id == 1
        assert session.total_frames == 1000
        assert session.fps == 30.0
        assert session.session_id is not None

    def test_creates_session_with_custom_fps(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000, fps=60.0)
        assert session.fps == 60.0

    def test_default_fps_from_service(self):
        svc = VideoReviewService(default_fps=25.0)
        session = svc.create_session(match_id=2, total_frames=500)
        assert session.fps == 25.0


class TestGetSession:
    def test_get_existing_session(self):
        svc = VideoReviewService()
        created = svc.create_session(match_id=1, total_frames=100)
        found = svc.get_session(created.session_id)
        assert found is created

    def test_get_nonexistent_session_returns_none(self):
        svc = VideoReviewService()
        assert svc.get_session("nonexistent") is None


class TestAddClip:
    def test_adds_clip_to_session(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "Test Clip", 100, 200)
        assert isinstance(clip, Clip)
        assert clip.title == "Test Clip"
        assert clip.start_frame == 100
        assert clip.end_frame == 200
        assert len(session.clips) == 1

    def test_clip_swaps_frames_when_end_before_start(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "Swapped", 300, 100)
        assert clip.start_frame == 100
        assert clip.end_frame == 300

    def test_clip_has_tags(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "Tagged", 0, 100, tags=["goal", "transition"])
        assert "goal" in clip.tags
        assert "transition" in clip.tags

    def test_add_clip_to_nonexistent_session(self):
        svc = VideoReviewService()
        clip = svc.add_clip("bad_id", "Test", 0, 100)
        assert clip is None

    def test_timestamps_from_fps(self):
        svc = VideoReviewService(default_fps=25.0)
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "Timed", 0, 50)
        assert clip.start_ts == 0.0
        assert clip.end_ts == 2.0


class TestAddAnnotation:
    def test_adds_annotation_to_session(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        ann = svc.add_annotation(
            session.session_id,
            AnnotationKind.ARROW,
            50,
            {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        )
        assert isinstance(ann, Annotation)
        assert ann.kind == AnnotationKind.ARROW
        assert ann.frame_number == 50
        assert len(session.annotations) == 1

    def test_add_annotation_to_nonexistent_session(self):
        svc = VideoReviewService()
        ann = svc.add_annotation("bad_id", AnnotationKind.CIRCLE, 0, {})
        assert ann is None

    def test_add_annotation_to_clip(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "Clip", 0, 100)
        ann = svc.add_annotation(
            session.session_id,
            AnnotationKind.TEXT,
            25,
            {"text": "hello"},
            clip_id=clip.clip_id,
        )
        assert ann in clip.annotations

    def test_timestamp_from_fps(self):
        svc = VideoReviewService(default_fps=25.0)
        session = svc.create_session(match_id=1, total_frames=1000)
        ann = svc.add_annotation(
            session.session_id,
            AnnotationKind.CIRCLE,
            100,
            {"r": 10},
        )
        assert ann.timestamp_s == 4.0


class TestRemoveClip:
    def test_removes_clip(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "To Remove", 0, 100)
        assert svc.remove_clip(session.session_id, clip.clip_id) is True
        assert len(session.clips) == 0

    def test_remove_nonexistent_session(self):
        svc = VideoReviewService()
        assert svc.remove_clip("bad_id", "clip_id") is False

    def test_remove_nonexistent_clip(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        assert svc.remove_clip(session.session_id, "no_such_clip") is False


class TestRemoveAnnotation:
    def test_removes_annotation_from_session(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        ann = svc.add_annotation(session.session_id, AnnotationKind.ARROW, 0, {})
        assert svc.remove_annotation(session.session_id, ann.annotation_id) is True
        assert len(session.annotations) == 0

    def test_removes_annotation_from_clip(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "C", 0, 100)
        ann = svc.add_annotation(
            session.session_id,
            AnnotationKind.CIRCLE,
            10,
            {},
            clip_id=clip.clip_id,
        )
        svc.remove_annotation(session.session_id, ann.annotation_id)
        assert ann not in clip.annotations

    def test_remove_nonexistent_session(self):
        svc = VideoReviewService()
        assert svc.remove_annotation("bad_id", "ann_id") is False


class TestFindClipsByTag:
    def test_finds_clips_by_tag(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        svc.add_clip(session.session_id, "A", 0, 100, tags=["goal"])
        svc.add_clip(session.session_id, "B", 100, 200, tags=["transition"])
        svc.add_clip(session.session_id, "C", 200, 300, tags=["goal", "set_piece"])
        results = svc.find_clips_by_tag(session.session_id, "goal")
        assert len(results) == 2
        assert all("goal" in c.tags for c in results)

    def test_find_no_matches(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        svc.add_clip(session.session_id, "A", 0, 100, tags=["goal"])
        results = svc.find_clips_by_tag(session.session_id, "nonexistent")
        assert len(results) == 0

    def test_find_on_nonexistent_session(self):
        svc = VideoReviewService()
        assert svc.find_clips_by_tag("bad_id", "goal") == []


class TestExportSession:
    def test_export_returns_dict(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        svc.add_clip(session.session_id, "Clip1", 0, 100, tags=["goal"])
        data = svc.export_session(session.session_id)
        assert data is not None
        assert data["match_id"] == 1
        assert len(data["clips"]) == 1
        assert data["clips"][0]["title"] == "Clip1"

    def test_export_with_annotation_references(self):
        svc = VideoReviewService()
        session = svc.create_session(match_id=1, total_frames=1000)
        clip = svc.add_clip(session.session_id, "C", 0, 100)
        ann = svc.add_annotation(
            session.session_id,
            AnnotationKind.ARROW,
            50,
            {},
            clip_id=clip.clip_id,
        )
        data = svc.export_session(session.session_id)
        assert ann.annotation_id in data["clips"][0]["annotations"]

    def test_export_nonexistent_session(self):
        svc = VideoReviewService()
        assert svc.export_session("bad_id") is None


class TestImportSession:
    def test_import_valid_payload(self):
        svc = VideoReviewService()
        payload = {
            "session_id": "test-import-1",
            "match_id": "2",
            "total_frames": "500",
            "fps": "30.0",
            "notes": "Imported session",
            "clips": [
                {
                    "clip_id": "clip-1",
                    "title": "Imported Clip",
                    "start_frame": "0",
                    "end_frame": "100",
                    "start_ts": "0.0",
                    "end_ts": "3.33",
                    "tags": ["goal"],
                    "notes": "",
                    "author": "",
                }
            ],
            "annotations": [
                {
                    "annotation_id": "ann-1",
                    "kind": "arrow",
                    "frame_number": "50",
                    "timestamp_s": "1.67",
                    "geometry": {"x1": 10, "y1": 20},
                    "color": "#FF0000",
                    "text": "",
                    "author": "",
                }
            ],
        }
        session = svc.import_session(payload)
        assert session is not None
        assert session.session_id == "test-import-1"
        assert session.match_id == 2
        assert len(session.clips) == 1
        assert len(session.annotations) == 1
        assert session.clips[0].title == "Imported Clip"
        assert session.annotations[0].kind == AnnotationKind.ARROW

    def test_import_invalid_payload_returns_none(self):
        svc = VideoReviewService()
        result = svc.import_session({"bad": "data"})
        assert result is None

    def test_import_payload_registered_in_sessions(self):
        svc = VideoReviewService()
        payload = {
            "session_id": "test-register",
            "match_id": "1",
            "total_frames": "100",
            "fps": "25.0",
            "notes": "",
            "clips": [],
            "annotations": [],
        }
        svc.import_session(payload)
        assert svc.get_session("test-register") is not None


class TestAvailable:
    def test_available_defaults_true(self):
        svc = VideoReviewService()
        assert svc.available is True
