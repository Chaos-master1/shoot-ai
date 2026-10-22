"""Tests for VRAMManager — GPU memory coordination."""

from __future__ import annotations

import sys
import types
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


# ---------------------------------------------------------------------------
# Torch stub (not available in test env)
# ---------------------------------------------------------------------------


def _install_torch_stub():
    if "torch" in sys.modules:
        return
    torch_mod = types.ModuleType("torch")

    class _Cuda:
        is_available = MagicMock(return_value=False)
        empty_cache = MagicMock()
        synchronize = MagicMock()

        class _DeviceProps:
            total_memory = 0

        get_device_properties = MagicMock(return_value=_DeviceProps())
        memory_allocated = MagicMock(return_value=0)

    torch_mod.cuda = _Cuda()
    sys.modules["torch"] = torch_mod


_install_torch_stub()

_mod = load_service_module("vram_test", "vram_manager.py")

ModelPriority = _mod.ModelPriority
VRAMStats = _mod.VRAMStats
VRAMManager = _mod.VRAMManager


class TestModelPriority:
    def test_enum_values(self):
        assert ModelPriority.YOLO.value == 100
        assert ModelPriority.LLM.value == 50
        assert ModelPriority.WHISPER.value == 25
        assert ModelPriority.OTHER.value == 10

    def test_enum_members(self):
        assert set(ModelPriority.__members__) == {"YOLO", "LLM", "WHISPER", "OTHER"}


class TestVRAMStats:
    def test_dataclass_defaults(self):
        s = VRAMStats(total_gb=10.0, used_gb=3.0, free_gb=7.0, percent_used=30.0)
        assert s.model_loaded is None

    def test_dataclass_with_model(self):
        s = VRAMStats(
            total_gb=10.0, used_gb=3.0, free_gb=7.0, percent_used=30.0, model_loaded="yolo"
        )
        assert s.model_loaded == "yolo"

    def test_dataclass_asdict(self):
        s = VRAMStats(1, 2, 3, 4, None)
        d = asdict(s)
        assert d["total_gb"] == 1
        assert d["used_gb"] == 2
        assert d["free_gb"] == 3
        assert d["percent_used"] == 4
        assert d["model_loaded"] is None


class TestVRAMManagerInit:
    def test_defaults(self):
        mgr = VRAMManager()
        assert mgr.total_budget == 10.0
        assert mgr.safety_margin == 1.0
        assert mgr._loaded_model is None
        assert mgr._loaded_objects == {}

    def test_custom_budget(self):
        mgr = VRAMManager(total_budget_gb=8.0, safety_margin_gb=0.5)
        assert mgr.total_budget == 8.0
        assert mgr.safety_margin == 0.5


class TestGetStats:
    def test_when_torch_not_available(self):
        with patch.dict("sys.modules", {"torch": None}):
            mgr = VRAMManager()
            stats = mgr.get_stats()
        assert stats.total_gb == 0
        assert stats.used_gb == 0
        assert stats.free_gb == 0
        assert stats.model_loaded is None

    def test_with_cuda_available(self):
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.get_device_properties = MagicMock(return_value=MagicMock(total_memory=12e9))
        torch.cuda.memory_allocated = MagicMock(return_value=4e9)
        mgr = VRAMManager()
        stats = mgr.get_stats()
        assert stats.total_gb == 12.0
        assert stats.used_gb == 4.0
        assert stats.free_gb == 8.0
        assert stats.percent_used == pytest.approx(33.33, rel=0.01)
        assert stats.model_loaded is None

    def test_tracks_loaded_model(self):
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.get_device_properties = MagicMock(return_value=MagicMock(total_memory=12e9))
        torch.cuda.memory_allocated = MagicMock(return_value=4e9)
        mgr = VRAMManager()
        mgr._loaded_model = "yolo"
        stats = mgr.get_stats()
        assert stats.model_loaded == "yolo"

    def test_get_stats_exception_returns_zero(self):
        import torch

        torch.cuda.is_available = MagicMock(side_effect=RuntimeError("CUDA error"))
        mgr = VRAMManager()
        stats = mgr.get_stats()
        assert stats.total_gb == 0


class TestHasRoomFor:
    def test_sufficient_room(self):
        mgr = VRAMManager()
        with patch.object(mgr, "get_stats", return_value=VRAMStats(10, 2, 8, 25, None)):
            assert mgr.has_room_for(3.0) is True

    def test_insufficient_room(self):
        mgr = VRAMManager()
        with patch.object(mgr, "get_stats", return_value=VRAMStats(10, 8, 2, 80, None)):
            assert mgr.has_room_for(3.0) is False

    def test_edge_boundary(self):
        mgr = VRAMManager(safety_margin_gb=1.0)
        with patch.object(mgr, "get_stats", return_value=VRAMStats(10, 5, 5, 50, None)):
            assert mgr.has_room_for(4.0) is True
            assert mgr.has_room_for(4.1) is False


class TestFree:
    def test_free_clears_state(self):
        mgr = VRAMManager()
        mgr._loaded_model = "yolo"
        mgr._loaded_objects["model"] = object()
        mgr.free()
        assert mgr._loaded_model is None
        assert mgr._loaded_objects == {}

    def test_free_when_nothing_loaded(self):
        mgr = VRAMManager()
        mgr.free()
        assert mgr._loaded_model is None

    @patch("gc.collect")
    def test_free_calls_gc(self, mock_gc):
        mgr = VRAMManager()
        mgr.free()
        mock_gc.assert_called_once()

    def test_free_clears_cuda_cache(self):
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.empty_cache = MagicMock()
        torch.cuda.synchronize = MagicMock()
        mgr = VRAMManager()
        mgr.free()
        torch.cuda.empty_cache.assert_called_once()
        torch.cuda.synchronize.assert_called_once()


class TestAllocateForYOLO:
    def test_success(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=True):
            result = mgr.allocate_for_yolo()
        assert result is True
        assert mgr._loaded_model == "yolo"

    def test_already_loaded(self):
        mgr = VRAMManager()
        mgr._loaded_model = "yolo"
        result = mgr.allocate_for_yolo()
        assert result is True

    def test_insufficient_vram(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=False):
            result = mgr.allocate_for_yolo()
        assert result is False
        assert mgr._loaded_model is None

    def test_custom_name_pass_through(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=True):
            result = mgr.allocate_for_yolo(model_name="yolo11x")
        assert result is True
        assert mgr._loaded_model == "yolo"


class TestAllocateForLLM:
    def test_success(self):
        mgr = VRAMManager()
        with patch.object(mgr, "get_stats", return_value=VRAMStats(12, 1, 11, 8.33, None)):
            result = mgr.allocate_for_llm()
        assert result is True
        assert mgr._loaded_model == "llm"

    def test_already_loaded(self):
        mgr = VRAMManager()
        mgr._loaded_model = "llm"
        result = mgr.allocate_for_llm()
        assert result is True

    def test_insufficient_vram(self):
        mgr = VRAMManager()
        with patch.object(mgr, "get_stats", return_value=VRAMStats(12, 10, 2, 83.33, None)):
            result = mgr.allocate_for_llm()
        assert result is False


class TestAllocateForWhisper:
    def test_success(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=True):
            result = mgr.allocate_for_whisper()
        assert result is True
        assert mgr._loaded_model == "whisper"

    def test_already_loaded(self):
        mgr = VRAMManager()
        mgr._loaded_model = "whisper"
        assert mgr.allocate_for_whisper() is True

    def test_insufficient_vram(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=False):
            result = mgr.allocate_for_whisper()
        assert result is False


class TestGetCurrentModel:
    def test_none_initially(self):
        mgr = VRAMManager()
        assert mgr.get_current_model() is None

    def test_after_allocation(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=True):
            mgr.allocate_for_yolo()
        assert mgr.get_current_model() == "yolo"

    def test_after_free(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=True):
            mgr.allocate_for_yolo()
        mgr.free()
        assert mgr.get_current_model() is None


class TestAllocationTransitions:
    def test_allocate_different_model_frees_previous(self):
        mgr = VRAMManager()
        with patch.object(mgr, "has_room_for", return_value=True):
            mgr.allocate_for_yolo()
            mgr.allocate_for_whisper()
        assert mgr._loaded_model == "whisper"

    def test_llm_frees_yolo(self):
        mgr = VRAMManager()
        with (
            patch.object(mgr, "has_room_for", return_value=True),
            patch.object(mgr, "get_stats", return_value=VRAMStats(12, 1, 11, 8.33, None)),
        ):
            mgr.allocate_for_yolo()
            mgr.allocate_for_llm()
        assert mgr._loaded_model == "llm"
