"""Tests for Phase 6 Sprint 3 — Telestration Complete + Highlight Reel Builder."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_tel_mod = load_service_module("telestration_test", "telestration_service.py")
TelestrationService = _tel_mod.TelestrationService

_hl_mod = load_service_module("highlight_reel_test", "highlight_reel_service.py")
HighlightReelService = _hl_mod.HighlightReelService
ReelClip = _hl_mod.ReelClip


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def tel_service():
    with tempfile.TemporaryDirectory() as d:
        svc = TelestrationService(presets_dir=d)
        yield svc


@pytest.fixture
def hl_service():
    with tempfile.TemporaryDirectory() as d:
        svc = HighlightReelService(output_dir=d)
        yield svc


@pytest.fixture
def sample_layers():
    return [
        {
            "id": "l1",
            "name": "Attack",
            "visible": True,
            "locked": False,
            "opacity": 1.0,
            "elements": [],
        },
        {
            "id": "l2",
            "name": "Defense",
            "visible": False,
            "locked": True,
            "opacity": 0.5,
            "elements": [],
        },
        {
            "id": "l3",
            "name": "Midfield",
            "visible": True,
            "locked": False,
            "opacity": 0.8,
            "elements": [{"type": "arrow", "x": 10, "y": 20}],
        },
    ]


@pytest.fixture
def mock_bridge():
    b = MagicMock()
    b._reel_output_dir = tempfile.gettempdir()
    return b


# ═══════════════════════════════════════════════════════════════════
# Telestration Layer Tests (8)
# ═══════════════════════════════════════════════════════════════════


class TestTelestrationLayers:
    def test_add_layer(self, tel_service):
        result = json.loads(tel_service.add_layer("layer_1", "Attack"))
        assert result["ok"] is True
        assert result["layer_id"] == "layer_1"

    def test_add_duplicate_layer(self, tel_service):
        tel_service.add_layer("layer_1", "Attack")
        result = json.loads(tel_service.add_layer("layer_1", "Dupe"))
        assert "error" in result

    def test_remove_layer(self, tel_service):
        tel_service.add_layer("layer_1", "Attack")
        result = json.loads(tel_service.remove_layer("layer_1"))
        assert result["ok"] is True

    def test_remove_nonexistent_layer(self, tel_service):
        result = json.loads(tel_service.remove_layer("nonexistent"))
        assert result["ok"] is True

    def test_toggle_layer_visibility(self, tel_service):
        tel_service.add_layer("layer_1", "Attack")
        result = json.loads(tel_service.toggle_layer_visibility("layer_1"))
        assert result["ok"] is True
        assert result["visible"] is False
        result2 = json.loads(tel_service.toggle_layer_visibility("layer_1"))
        assert result2["visible"] is True

    def test_toggle_nonexistent_layer(self, tel_service):
        result = json.loads(tel_service.toggle_layer_visibility("missing"))
        assert "error" in result

    def test_set_layer_opacity(self, tel_service):
        tel_service.add_layer("layer_1", "Attack")
        result = json.loads(tel_service.set_layer_opacity("layer_1", 0.5))
        assert result["ok"] is True
        assert result["opacity"] == 0.5

    def test_set_layer_opacity_clamps(self, tel_service):
        tel_service.add_layer("layer_1", "Attack")
        result = json.loads(tel_service.set_layer_opacity("layer_1", 1.5))
        assert result["opacity"] == 1.0
        result2 = json.loads(tel_service.set_layer_opacity("layer_1", -0.5))
        assert result2["opacity"] == 0.0

    def test_get_layers_empty(self, tel_service):
        result = json.loads(tel_service.get_layers())
        assert result["layers"] == []

    def test_get_layers_with_data(self, tel_service):
        tel_service.add_layer("l1", "Attack")
        tel_service.add_layer("l2", "Defense")
        result = json.loads(tel_service.get_layers())
        assert len(result["layers"]) == 2
        names = [layer["name"] for layer in result["layers"]]
        assert "Attack" in names
        assert "Defense" in names


# ═══════════════════════════════════════════════════════════════════
# Telestration Preset Tests (6)
# ═══════════════════════════════════════════════════════════════════


class TestTelestrationPresets:
    def test_save_preset(self, tel_service):
        layers = [
            {
                "id": "l1",
                "name": "Attack",
                "visible": True,
                "locked": False,
                "opacity": 1.0,
                "elements": [],
            }
        ]
        result = json.loads(tel_service.save_preset("test_preset", json.dumps(layers)))
        assert result["ok"] is True
        assert result["preset"] == "test_preset"

    def test_list_presets_empty(self, tel_service):
        result = json.loads(tel_service.list_presets())
        assert result["presets"] == []

    def test_list_presets_after_save(self, tel_service):
        layers = [
            {
                "id": "l1",
                "name": "Attack",
                "visible": True,
                "locked": False,
                "opacity": 1.0,
                "elements": [],
            }
        ]
        tel_service.save_preset("preset_a", json.dumps(layers))
        tel_service.save_preset("preset_b", json.dumps(layers))
        result = json.loads(tel_service.list_presets())
        assert len(result["presets"]) == 2

    def test_load_preset(self, tel_service):
        layers = [
            {
                "id": "l1",
                "name": "Attack",
                "visible": True,
                "locked": False,
                "opacity": 1.0,
                "elements": [],
            }
        ]
        tel_service.save_preset("test_preset", json.dumps(layers))
        result = json.loads(tel_service.load_preset("test_preset"))
        assert result["ok"] is True
        assert result["preset"] == "test_preset"

    def test_load_nonexistent_preset(self, tel_service):
        result = json.loads(tel_service.load_preset("missing"))
        assert "error" in result

    def test_delete_preset(self, tel_service):
        layers = [
            {
                "id": "l1",
                "name": "Attack",
                "visible": True,
                "locked": False,
                "opacity": 1.0,
                "elements": [],
            }
        ]
        tel_service.save_preset("test_preset", json.dumps(layers))
        result = json.loads(tel_service.delete_preset("test_preset"))
        assert result["ok"] is True
        list_result = json.loads(tel_service.list_presets())
        assert len(list_result["presets"]) == 0


# ═══════════════════════════════════════════════════════════════════
# Telestration Export Tests (4)
# ═══════════════════════════════════════════════════════════════════


class TestTelestrationExport:
    def test_export_empty_layers(self, tel_service):
        result = json.loads(tel_service.export_annotated_video("dummy.mp4", "[]"))
        assert "error" in result

    def test_export_nonexistent_video(self, tel_service):
        layers = [{"id": "l1", "name": "Export", "elements": [{"type": "text", "text": "Test"}]}]
        result = json.loads(
            tel_service.export_annotated_video("/nonexistent/video.mp4", json.dumps(layers))
        )
        assert "error" in result

    def test_export_with_text_elements(self, tel_service):
        layers = [
            {
                "id": "l1",
                "name": "Export",
                "elements": [{"type": "text", "text": "Test", "x": 10, "y": 20}],
            }
        ]
        # No actual video, should fail gracefully
        result = json.loads(
            tel_service.export_annotated_video("/tmp/nonexistent.mp4", json.dumps(layers))
        )
        assert "error" in result

    def test_export_no_drawable_elements(self, tel_service):
        layers = [
            {"id": "l1", "name": "Export", "elements": [{"type": "circle", "x": 10, "y": 20}]}
        ]
        result = json.loads(
            tel_service.export_annotated_video("/tmp/nonexistent.mp4", json.dumps(layers))
        )
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════
# Highlight Reel Service Tests (8)
# ═══════════════════════════════════════════════════════════════════


class TestHighlightReelService:
    def test_make_reel_from_events_basic(self, hl_service):
        events = [{"timestamp": 10.0, "type": "goal"}, {"timestamp": 30.0, "type": "shot"}]
        result = json.loads(hl_service.make_reel_from_events(1, events, "/tmp/dummy.mp4"))
        assert result["clip_count"] == 2
        assert result["clips_defined"] is True

    def test_make_reel_from_events_empty(self, hl_service):
        result = json.loads(hl_service.make_reel_from_events(1, [], "/tmp/dummy.mp4"))
        assert result["clip_count"] == 0

    def test_make_reel_context_window(self, hl_service):
        events = [{"timestamp": 10.0, "type": "goal"}]
        result = json.loads(
            hl_service.make_reel_from_events(1, events, "/tmp/dummy.mp4", context_seconds=5.0)
        )
        assert result["clip_count"] == 1

    def test_compose_reel_empty_clips(self, hl_service):
        import asyncio

        result = json.loads(asyncio.run(hl_service.compose_reel([], "empty.mp4")))
        assert "error" in result

    def test_compose_reel_invalid_clips(self, hl_service):
        clips = [
            ReelClip(video_path="C:\\nonexistent_video_test.mp4", start_seconds=0, end_seconds=10)
        ]
        import asyncio

        with pytest.raises((ValueError, RuntimeError)):
            asyncio.run(hl_service.compose_reel(clips, "invalid.mp4"))

    def test_reel_clip_duration_validation(self, hl_service):
        clip = ReelClip(video_path="/tmp/dummy.mp4", start_seconds=10, end_seconds=5)
        assert clip.end_seconds <= clip.start_seconds

    def test_reel_clip_labels(self, hl_service):
        clip = ReelClip(video_path="/tmp/dummy.mp4", start_seconds=0, end_seconds=10, label="goal")
        assert clip.label == "goal"

    def test_reel_result_dataclass(self):
        result = _hl_mod.ReelResult(
            output_path="/tmp/reel.mp4", clip_count=3, total_duration_seconds=15.0
        )
        assert result.output_path == "/tmp/reel.mp4"
        assert result.clip_count == 3
        assert result.total_duration_seconds == 15.0


# ═══════════════════════════════════════════════════════════════════
# Reel Progress Bridge Tests (4)
# ═══════════════════════════════════════════════════════════════════


class TestReelProgress:
    def test_reel_status_starting(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        # Manually set progress
        handler._reel_progress["test_1"] = {
            "status": "starting",
            "progress": 0.0,
            "output_path": "",
        }
        result = json.loads(handler.reel_status("test_1"))
        assert result["status"] == "starting"
        assert result["progress"] == 0.0

    def test_reel_status_processing(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        handler._reel_progress["test_2"] = {
            "status": "processing",
            "progress": 0.5,
            "output_path": "",
        }
        result = json.loads(handler.reel_status("test_2"))
        assert result["status"] == "processing"
        assert result["progress"] == 0.5

    def test_reel_status_complete(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        handler._reel_progress["test_3"] = {
            "status": "complete",
            "progress": 1.0,
            "output_path": "/tmp/reel.mp4",
        }
        result = json.loads(handler.reel_status("test_3"))
        assert result["status"] == "complete"
        assert result["progress"] == 1.0
        assert result["output_path"] == "/tmp/reel.mp4"

    def test_reel_status_unknown_id(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        result = json.loads(handler.reel_status("nonexistent"))
        assert result["status"] == "unknown"
        assert result["progress"] == 0.0

    def test_reel_status_error(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        handler._reel_progress["test_err"] = {"status": "error", "progress": 0.0, "output_path": ""}
        result = json.loads(handler.reel_status("test_err"))
        assert result["status"] == "error"

    def test_reel_from_events_with_progress(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        events = [{"timestamp": 10.0, "type": "goal"}, {"timestamp": 30.0, "type": "shot"}]
        result_str = handler.reel_from_events(1, json.dumps(events), "/tmp/dummy.mp4")
        result = json.loads(result_str)
        assert "reel_id" in result
        # Check reel_id was tracked
        status = json.loads(handler.reel_status(result["reel_id"]))
        assert status["status"] in ("complete", "error")


# ═══════════════════════════════════════════════════════════════════
# VideoHandler reel_from_events integration (2)
# ═══════════════════════════════════════════════════════════════════


class TestVideoHandlerReel:
    def test_reel_from_events_success_path(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        events = [{"timestamp": 5.0, "type": "goal"}, {"timestamp": 15.0, "type": "shot"}]
        result = json.loads(handler.reel_from_events(1, json.dumps(events), "/tmp/test.mp4"))
        assert "reel_id" in result
        assert result["clips_defined"] is True

    def test_reel_from_events_invalid_json(self, mock_bridge):
        from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

        handler = VideoHandler(mock_bridge, {})
        result = json.loads(handler.reel_from_events(1, "{invalid", "/tmp/test.mp4"))
        assert "error" in result
