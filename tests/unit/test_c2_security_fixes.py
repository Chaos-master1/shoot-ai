"""Regression tests for the Phase-C2 security fixes.

Pins two behaviors so a revert fails loudly:

1. ``reel_compose`` must reject traversal/escape in the user-supplied
   ``output_filename`` -- it is joined onto the highlight-reel service
   output dir, so ``../x.mp4`` would write outside it (same injection
   class previously hardened in ``stream_start_capture``).

2. Vendor-import handlers must validate user-supplied paths BEFORE
   handing them to parsers: documents-directory allowlist (traversal
   denial) plus the data-extension allowlist (.json/.xml/.csv).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from conftest import install_kawkab_stubs

install_kawkab_stubs()

# Handlers are built through the real __init__ (real RateLimiter by default)
# rather than __new__+attribute pokes: hand-setting _rate_limiter=None used to
# silently disable limiting, hiding exactly the bug class the BridgeHandlerBase
# refactor exists to prevent.

from kawkab.ui.bridge_handlers.bridge_import import (  # noqa: E402
    ImportHandler,
)
from kawkab.ui.bridge_handlers.bridge_video import (  # noqa: E402
    VideoHandler,
)


def _video_handler() -> VideoHandler:
    return VideoHandler(None, {"storage_service": MagicMock()})


def _import_handler() -> ImportHandler:
    return ImportHandler(None, {"storage_service": MagicMock()})


class TestReelComposeFilenameValidation:
    """The output filename is joined onto the service output dir."""

    def test_rejects_parent_traversal(self):
        out = _video_handler().reel_compose("[]", "../escape.mp4")
        assert "invalid output filename" in out

    def test_rejects_path_separators(self):
        h = _video_handler()
        for bad in ("sub/dir/reel.mp4", "back\\slash.mp4", "a..b.mp4"):
            assert "invalid output filename" in h.reel_compose("[]", bad), bad

    def test_rejects_non_mp4_and_empty(self):
        h = _video_handler()
        assert "invalid output filename" in h.reel_compose("[]", "reel.avi")
        assert "invalid output filename" in h.reel_compose("[]", "")

    def test_valid_filename_passes_validation_gate(self):
        """A clean .mp4 name must get past the filename gate (it then hits
        the real ffmpeg compose, which is not exercised here -- the mocked
        service stands in)."""
        h = _video_handler()
        h._highlight_reel = MagicMock()
        h._highlight_reel.compose_reel = MagicMock(return_value='{"ok": true}')
        out = h.reel_compose("[]", "my_reel.mp4")
        h._highlight_reel.compose_reel.assert_called_once()
        assert "invalid output filename" not in out


class TestVendorImportPathValidation:
    """Vendor feeds are user-supplied paths -- validate before parsing.

    The parsing service is mocked to always succeed: any ``success`` in
    the handler response then proves the input passed validation. On
    pre-fix code these tests FAIL (the mock runs -- the handler never
    validated), which is the discrimination we want.
    """

    def _patch_tracking_service(self):
        import kawkab.services.vendor_tracking_import_service as vtm

        original = vtm.VendorTrackingImportService

        class _FakeSvc:
            def __init__(self, storage):
                pass

            async def import_tracking_file(self, path, **kw):
                return {"frames_imported": 1}

        vtm.VendorTrackingImportService = _FakeSvc
        return vtm, original

    def _patch_event_service(self):
        import kawkab.services.vendor_event_import_service as vei

        original = vei.VendorEventImportService

        class _FakeSvc:
            def __init__(self, storage):
                pass

            async def import_opta_f24(self, path, *a, **kw):
                return {"events_imported": 1}

            async def import_wyscout(self, path, **kw):
                return {"events_imported": 1}

        vei.VendorEventImportService = _FakeSvc
        return vei, original

    def test_tracking_import_rejects_traversal(self):
        vtm, original = self._patch_tracking_service()
        try:
            # Absolute path OUTSIDE the documents allowlist.
            out = __import__("asyncio").run(
                _import_handler().import_tracking_file("/etc/hostname.json")
            )
            assert json.loads(out)["success"] is False
            assert "traversal" in json.loads(out).get("error", "").lower()
        finally:
            vtm.VendorTrackingImportService = original

    def test_tracking_import_rejects_bad_extension(self):
        vtm, original = self._patch_tracking_service()
        try:
            out = __import__("asyncio").run(_import_handler().import_tracking_file("feed.exe"))
            assert json.loads(out)["success"] is False
            assert "unsupported" in json.loads(out).get("error", "").lower()
        finally:
            vtm.VendorTrackingImportService = original

    def test_event_import_rejects_bad_extension(self):
        vei, original = self._patch_event_service()
        try:
            out = __import__("asyncio").run(_import_handler().import_event_file("malware.exe"))
            assert json.loads(out)["success"] is False
            assert "unsupported" in json.loads(out).get("error", "").lower()
        finally:
            vei.VendorEventImportService = original

    def test_event_import_rejects_f7_traversal(self):
        from kawkab.core.paths import get_paths

        docs = get_paths().documents
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "ok_feed.xml").write_text("<x/>", encoding="utf-8")
        vei, original = self._patch_event_service()
        try:
            out = __import__("asyncio").run(
                _import_handler().import_event_file(
                    str(docs / "ok_feed.xml"), f7_path="/etc/hostname.xml"
                )
            )
            assert json.loads(out)["success"] is False
        finally:
            vei.VendorEventImportService = original

    def test_tracking_import_accepts_documents_file(self):
        """A .json inside the allowlisted documents dir must pass path
        validation (the mocked storage stands in for the parser) -- guards
        against over-blocking legitimate imports."""
        from kawkab.core.paths import get_paths

        docs = get_paths().documents
        docs.mkdir(parents=True, exist_ok=True)
        ok_file = docs / "valid_feed.json"
        ok_file.write_text("{}", encoding="utf-8")

        h = _import_handler()
        captured = {}
        vtm, original = self._patch_tracking_service()

        class _Capturing(_FakeSvc := vtm.VendorTrackingImportService):  # noqa: F841
            pass

        orig_init = original

        class _FakeCapture:
            def __init__(self, storage):
                pass

            async def import_tracking_file(self, path, **kw):
                captured["path"] = path
                return {"frames_imported": 1}

        vtm.VendorTrackingImportService = _FakeCapture
        try:
            out = __import__("asyncio").run(h.import_tracking_file(str(ok_file)))
        finally:
            vtm.VendorTrackingImportService = orig_init
        assert json.loads(out)["success"] is True
        assert captured["path"] == str(ok_file)
