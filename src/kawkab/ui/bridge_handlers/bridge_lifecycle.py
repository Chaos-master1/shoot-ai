"""Handler for app lifecycle bridge methods (getAppState, getGPUStatus, profiler, metrics)."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.observability import metrics
from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class LifecycleHandler(BridgeHandlerBase):
    """Handles app lifecycle operations for Bridge."""

    # ── helpers ──────────────────────────────────────────────────

    @property
    def benchmark_service(self):
        return self._services.get("benchmark_service")

    @property
    def cv_service(self):
        return self._services.get("cv_service")

    @property
    def profiler(self):
        return self._services.get("profiler")

    # ── slots ────────────────────────────────────────────────────

    def get_gpu_info(self):
        self._check_rate_limit()
        from kawkab.services.benchmark_service import BenchmarkService

        try:
            model_size = getattr(self.cv_service, "model_size", "l") if self.cv_service else "l"
            if self.benchmark_service is None:
                return json.dumps(
                    {
                        "gpu_name": "unknown",
                        "tier": "unknown",
                        "recommendations": BenchmarkService.recommend_settings("unknown"),
                        "current_settings": {
                            "model_size": model_size,
                            "frame_skip": self._services.get("frame_skip", 3),
                        },
                    }
                )
            info = self.benchmark_service._system_info
            gpu_name = info.get("gpu_name", "unknown")
            tier = BenchmarkService.classify_gpu_tier(gpu_name)
            recommendations = BenchmarkService.recommend_settings(tier)
            return json.dumps(
                {
                    "gpu_name": gpu_name,
                    "tier": tier,
                    "recommendations": recommendations,
                    "current_settings": {
                        "model_size": model_size,
                        "frame_skip": self._services.get("frame_skip", 3),
                    },
                }
            )
        except Exception as e:
            logger.error(f"get_gpu_info failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def profiler_status(self):
        self._check_rate_limit()
        if self.profiler is None:
            return json.dumps({"error": "Profiler not initialized"})
        try:
            return json.dumps(self.profiler.report().to_dict())
        except Exception as e:
            logger.error(f"profiler_status failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def profiler_reset(self):
        self._check_rate_limit()
        if self.profiler is None:
            return json.dumps({"error": "Profiler not initialized"})
        try:
            self.profiler.reset()
            self.profiler.start()
            return json.dumps({"ok": True, "message": "Profiler reset"})
        except Exception as e:
            logger.error(f"profiler_reset failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def metrics_text(self):
        self._check_rate_limit()
        return metrics.render()
