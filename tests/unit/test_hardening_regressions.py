"""Regression tests for the v0.13.1 hardening pass.

Pins three behaviors end to end:
1. scout_search_players never fabricates players -- an internal failure
   returns an honest empty result with an error field, and the source no
   longer contains the old hardcoded celebrity-players fallback.
2. stream_start_capture rejects file:// URLs and hostile output filenames
   before ffmpeg ever sees them.
3. import_wearable / upload_face_photo validate user-supplied paths through
   SecurityValidator before any parser or cv2 call touches them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawkab.core.security import SecurityValidator
from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler
from kawkab.ui.bridge_handlers.bridge_cloud import CloudCollabHandler
from kawkab.ui.bridge_handlers.bridge_recruitment import RecruitmentHandler


def _recruitment_handler() -> RecruitmentHandler:
    return RecruitmentHandler(bridge=None, services={}, rate_limiter=None)


def _analysis_handler() -> AnalysisHandler:
    return AnalysisHandler(bridge=None, services={}, rate_limiter=None)


def _cloud_handler() -> CloudCollabHandler:
    return CloudCollabHandler(bridge=None, services={}, rate_limiter=None)


def _point_documents_at(monkeypatch, docs):
    """Make get_paths().documents resolve to ``docs`` regardless of which
    paths module implementation is loaded.

    Patching env vars + singleton state is fragile here: tests/conftest.py
    stubs the module with _default_paths, the real module caches _paths,
    and some test modules write XDG_DOCUMENTS_DIR at import time and never
    restore it (test_vendor_api_endpoints.py), so a rebuilt singleton can
    inherit a poisoned env. The validators import get_paths *inside the
    function body*, so patching the module attribute directly is fully
    deterministic and monkeypatch-restorable.
    """
    import kawkab.core.paths as paths_mod

    class _FakePaths:
        documents = docs

    monkeypatch.setattr(paths_mod, "get_paths", lambda: _FakePaths())


class TestScoutSearchHonestFailure:
    """scout_search_players used to fall back to a hardcoded celebrity DB
    (Haaland/Messi/...) on any internal error -- a coaching tool must not
    invent stats a scout report would be briefed from."""

    @pytest.mark.asyncio
    async def test_internal_failure_returns_honest_empty_result(self):
        """Force the real-search branch to explode; result must be empty,
        not fabricated."""

        # A query whose inner machinery fails: the real branch imports
        # kawkab.core.player_search and opens a players.json relative to the
        # repo. Point the search at a malformed DB so the branch raises.
        handler = _recruitment_handler()
        import unittest.mock as mock

        with (
            mock.patch("builtins.open", side_effect=OSError("disk exploded")),
            mock.patch("os.path.exists", return_value=True),
        ):
            result = json.loads(await handler.scout_search_players("haaland"))

        assert result["results"] == []
        assert result["total"] == 0
        assert result.get("error"), "internal failure must surface an error field"

    def test_source_has_no_mock_player_fallback(self):
        src = Path("src/kawkab/ui/bridge_handlers/bridge_recruitment.py").read_text(
            encoding="utf-8"
        )
        assert "mock_db" not in src, "hardcoded player fallback is back in source"
        assert "Erling Haaland" not in src


class TestStreamCaptureRejections:
    """stream_start_capture hands the URL to `ffmpeg -i` and joins
    output_filename onto the capture dir -- both must be screened."""

    @pytest.mark.asyncio
    async def test_rejects_file_url(self):
        result = json.loads(await _cloud_handler().stream_start_capture("file:///etc/passwd"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_empty_url(self):
        result = json.loads(await _cloud_handler().stream_start_capture(""))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_filename_with_semicolon_injection(self):
        result = json.loads(
            await _cloud_handler().stream_start_capture(
                "rtmp://media/x", output_filename="a;rm -rf x.mp4"
            )
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_path_traversal_filename(self):
        # sanitize_string's default allowlist keeps '/' and '.', so the
        # explicit separator/.. guard is what rejects this one.
        result = json.loads(
            await _cloud_handler().stream_start_capture(
                "rtmp://media/x", output_filename="../../evil.mp4"
            )
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_accepts_clean_filename(self):
        result = json.loads(
            await _cloud_handler().stream_start_capture(
                "rtmp://localhost/live", output_filename="capture-2026.mp4"
            )
        )
        # The service itself will fail to spawn ffmpeg in CI (no stream),
        # but the request must pass validation -- i.e. the error, if any,
        # must come from the service layer, not the filename guard.
        assert "invalid output filename" not in json.dumps(result)


class TestUserPathValidation:
    """Handlers that open user-supplied paths must run SecurityValidator
    before any parser/cv2 call."""

    @pytest.mark.asyncio
    async def test_import_wearable_rejects_path_outside_allowlist(self):
        # Allowed wearable extension, but outside the documents allowlist ->
        # the traversal branch (not the extension branch) must reject it.
        result = json.loads(await _analysis_handler().import_wearable("/etc/passwd.gpx"))
        assert "error" in result
        assert "traversal" in result["error"].lower() or "not within" in result["error"]

    @pytest.mark.asyncio
    async def test_import_wearable_rejects_wrong_extension(self, tmp_path):
        f = tmp_path / "data.exe"
        f.write_text("MZ...")
        result = json.loads(await _analysis_handler().import_wearable(str(f)))
        assert "error" in result
        assert "Unsupported wearable file type" in result["error"]

    @pytest.mark.asyncio
    async def test_upload_face_photo_rejects_path_outside_allowlist(self):
        result = json.loads(await _analysis_handler().upload_face_photo("/etc/passwd", "Test", 9))
        assert "error" in result

    def test_validator_blocks_traversal_sequences(self):
        with pytest.raises(ValueError, match="[Tt]raversal|not within"):
            SecurityValidator.validate_data_file_path("../../etc/passwd.json")

    def test_validator_allows_documents_kawkabai_tree(self, tmp_path, monkeypatch):
        """A wearable file inside the documents allowlist validates; the same
        path outside it is denied."""
        docs = tmp_path / "xdg-docs" / "KawkabAI"
        docs.mkdir(parents=True)
        f = docs / "session.gpx"
        f.write_text("<gpx></gpx>")

        _point_documents_at(monkeypatch, docs)
        resolved = SecurityValidator.validate_wearable_path(str(f))
        assert resolved == f.resolve()

        outside = tmp_path / "elsewhere.gpx"
        outside.write_text("<gpx></gpx>")
        with pytest.raises(ValueError, match="[Tt]raversal|not within"):
            SecurityValidator.validate_wearable_path(str(outside))

    def test_image_validator_allows_real_photo_extensions(self, tmp_path, monkeypatch):
        """upload_face_photo must accept actual photos -- the generic data
        allowlist (.json/.xml/.csv) would have rejected every .jpg/.png."""
        docs = tmp_path / "xdg-docs" / "KawkabAI"
        docs.mkdir(parents=True)
        photo = docs / "face.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic

        _point_documents_at(monkeypatch, docs)
        assert SecurityValidator.validate_image_path(str(photo)) == photo.resolve()

        with pytest.raises(ValueError, match="Unsupported image"):
            SecurityValidator.validate_image_path(str(docs / "notes.txt"))
