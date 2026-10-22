"""VRAM Manager - coordinates GPU memory for multiple models.

With 12GB on RTX 4070, we can't run YOLOv11l + BoT-SORT + Qwen 2.5 14B
simultaneously. The VRAMManager ensures models are loaded only when needed
and freed after use.

Priority order:
1. YOLO (needed for video processing)
2. LLM (needed after CV for reports)
3. Whisper (optional, only if audio analysis enabled)

The manager:
- Tracks current VRAM usage
- Loads/unloads models on demand
- Falls back to CPU if VRAM insufficient
- Logs all transitions for debugging
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class ModelPriority(Enum):
    """Model loading priority (higher = loaded first)."""

    YOLO = 100
    LLM = 50
    WHISPER = 25
    OTHER = 10


@dataclass
class VRAMStats:
    """Current VRAM usage statistics."""

    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float
    model_loaded: str | None = None


class VRAMManager:
    """Manages GPU memory for multiple AI models.

    Usage:
        manager = VRAMManager(total_budget_gb=10.0)
        manager.allocate_for_yolo()  # Loads YOLO, unloads others
        ... do CV ...
        manager.release()  # Unloads YOLO
        manager.allocate_for_llm()  # Loads LLM
        ... do LLM ...
        manager.release()
    """

    def __init__(
        self,
        total_budget_gb: float = 10.0,
        safety_margin_gb: float = 1.0,
    ) -> None:
        self.total_budget = total_budget_gb
        self.safety_margin = safety_margin_gb
        self._loaded_model: str | None = None
        self._loaded_objects: dict[str, Any] = {}
        logger.info(f"VRAMManager: budget={total_budget_gb}GB, safety_margin={safety_margin_gb}GB")

    def get_stats(self) -> VRAMStats:
        """Get current VRAM usage."""
        try:
            import torch

            if not torch.cuda.is_available():
                return VRAMStats(0, 0, 0, 0, None)

            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            used = torch.cuda.memory_allocated(0) / 1e9
            free = total - used
            pct = (used / total) * 100 if total > 0 else 0
            return VRAMStats(
                total_gb=total,
                used_gb=used,
                free_gb=free,
                percent_used=pct,
                model_loaded=self._loaded_model,
            )
        except Exception as e:
            logger.debug(f"Could not get VRAM stats: {e}")
            return VRAMStats(0, 0, 0, 0, None)

    def has_room_for(self, model_gb: float) -> bool:
        """Check if there's room for a model of given size."""
        stats = self.get_stats()
        return stats.free_gb >= (model_gb + self.safety_margin)

    def free(self) -> None:
        """Free all loaded model memory."""
        logger.debug(f"Freeing {self._loaded_model or 'no'} model from VRAM")
        self._loaded_objects.clear()
        self._loaded_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        stats = self.get_stats()
        logger.info(f"VRAM after free: {stats.free_gb:.2f}GB free")

    def allocate_for_yolo(self, model_name: str = "yolo11l") -> bool:
        """Allocate VRAM for YOLO model. Returns True if successful."""
        if self._loaded_model == "yolo":
            return True

        self.free()

        yolo_gb = 4.0
        if not self.has_room_for(yolo_gb):
            logger.warning(
                f"Insufficient VRAM for YOLO "
                f"(need {yolo_gb}GB, "
                f"have {self.get_stats().free_gb:.1f}GB). Falling back to CPU."
            )
            return False

        logger.info(f"VRAM allocated for YOLO: {yolo_gb}GB reserved")
        self._loaded_model = "yolo"
        return True

    def allocate_for_llm(self, model_name: str = "qwen3:14b") -> bool:
        """Allocate VRAM for LLM. Returns True if successful.

        If VRAM insufficient, returns False. Caller should use CPU fallback
        or num_gpu=0 in Ollama config.
        """
        if self._loaded_model == "llm":
            return True

        self.free()

        llm_gb = 9.0
        stats = self.get_stats()
        if stats.free_gb < llm_gb + self.safety_margin:
            logger.warning(
                f"Insufficient VRAM for LLM "
                f"(need {llm_gb}GB, have {stats.free_gb:.1f}GB). "
                f"Use CPU fallback (num_gpu=0)."
            )
            return False

        logger.info(f"VRAM allocated for LLM: {llm_gb}GB reserved")
        self._loaded_model = "llm"
        return True

    def allocate_for_whisper(self, model_size: str = "base") -> bool:
        """Allocate VRAM for Whisper. Returns True if successful."""
        if self._loaded_model == "whisper":
            return True

        self.free()
        whisper_gb = 1.5

        if not self.has_room_for(whisper_gb):
            logger.warning(
                f"Insufficient VRAM for Whisper "
                f"(need {whisper_gb}GB, have {self.get_stats().free_gb:.1f}GB)"
            )
            return False

        logger.info(f"VRAM allocated for Whisper: {whisper_gb}GB")
        self._loaded_model = "whisper"
        return True

    def get_current_model(self) -> str | None:
        """Get the name of the currently loaded model."""
        return self._loaded_model
