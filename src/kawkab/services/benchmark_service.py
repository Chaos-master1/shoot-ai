"""Performance benchmarking service - measures analysis speed and resource usage.

Tracks per-stage performance for establishing baselines and optimizing
for different GPU tiers. Stores results in the database for trend analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

_BENCHMARK_CACHE_PATH = Path("data/benchmark_cache.json")


def _load_benchmark_cache() -> dict[str, Any]:
    """Load cached benchmark results keyed by GPU model."""
    if _BENCHMARK_CACHE_PATH.exists():
        try:
            return json.loads(_BENCHMARK_CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_benchmark_cache(cache: dict[str, Any]) -> None:
    """Save benchmark cache to disk."""
    try:
        _BENCHMARK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BENCHMARK_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save benchmark cache: {e}")


@dataclass
class BenchmarkResult:
    """Performance metrics for a single analysis run."""

    match_id: int | None = None
    video_path: str = ""
    video_duration_seconds: float = 0.0
    total_frames: int = 0

    # Overall
    total_time_seconds: float = 0.0
    realtime_ratio: float = 0.0  # 0.5 = 2x slower than realtime
    fps_effective: float = 0.0

    # Per-stage (seconds)
    stage_enhancement_seconds: float = 0.0
    stage_detection_seconds: float = 0.0
    stage_tracking_seconds: float = 0.0
    stage_analysis_seconds: float = 0.0
    stage_advanced_metrics_seconds: float = 0.0
    stage_save_seconds: float = 0.0

    # Resources
    peak_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    gpu_utilization_pct: float = 0.0

    # System
    gpu_name: str = "unknown"
    cpu_name: str = "unknown"
    ram_gb: float = 0.0
    model_size: str = "l"
    frame_skip: int = 3


class BenchmarkService:
    """Measures and records performance metrics for analysis runs."""

    def __init__(self) -> None:
        self._stage_times: dict[str, float] = {}
        self._start_times: dict[str, float] = {}
        self._peak_memory_mb = 0.0
        self._peak_gpu_memory_mb = 0.0
        self._process = psutil.Process()
        self._system_info = self._detect_system_info()
        logger.info(f"BenchmarkService initialized: {self._system_info}")

    def _detect_system_info(self) -> dict[str, Any]:
        """Detect system hardware information."""
        info = {
            "cpu_name": "unknown",
            "gpu_name": "unknown",
            "ram_gb": 0.0,
        }
        try:
            import platform

            info["cpu_name"] = platform.processor() or "unknown"
            info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception as e:
            logger.warning(f"Failed to detect CPU/RAM: {e}")

        try:
            import torch

            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning(f"Failed to detect GPU: {e}")

        return info

    def start_stage(self, stage_name: str) -> None:
        """Start timing a named stage."""
        self._start_times[stage_name] = time.perf_counter()
        self._update_memory_peak()

    def end_stage(self, stage_name: str) -> None:
        """End timing a named stage and record duration."""
        if stage_name not in self._start_times:
            logger.warning(f"Stage {stage_name} was never started")
            return
        elapsed = time.perf_counter() - self._start_times[stage_name]
        self._stage_times[stage_name] = elapsed
        self._update_memory_peak()
        logger.debug(f"Stage {stage_name}: {elapsed:.3f}s")

    def _update_memory_peak(self) -> None:
        """Update peak memory usage."""
        try:
            mem = self._process.memory_info().rss / (1024 * 1024)
            if mem > self._peak_memory_mb:
                self._peak_memory_mb = mem
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if gpu_mem > self._peak_gpu_memory_mb:
                    self._peak_gpu_memory_mb = gpu_mem
        except Exception as e:
            logger.debug(f"GPU memory check failed: {e}")

    def record_gpu_utilization(self, utilization_pct: float) -> None:
        """Record average GPU utilization for the run."""
        self._gpu_utilization = utilization_pct

    def build_result(
        self,
        match_id: int | None = None,
        video_path: str = "",
        video_duration_seconds: float = 0.0,
        total_frames: int = 0,
        model_size: str = "l",
        frame_skip: int = 3,
    ) -> BenchmarkResult:
        """Build the final benchmark result from recorded stages."""
        total_time = sum(self._stage_times.values())
        realtime_ratio = total_time / video_duration_seconds if video_duration_seconds > 0 else 0.0
        fps_effective = total_frames / total_time if total_time > 0 else 0.0

        result = BenchmarkResult(
            match_id=match_id,
            video_path=video_path,
            video_duration_seconds=video_duration_seconds,
            total_frames=total_frames,
            total_time_seconds=total_time,
            realtime_ratio=round(realtime_ratio, 3),
            fps_effective=round(fps_effective, 2),
            stage_enhancement_seconds=round(self._stage_times.get("enhancement", 0.0), 3),
            stage_detection_seconds=round(self._stage_times.get("detection", 0.0), 3),
            stage_tracking_seconds=round(self._stage_times.get("tracking", 0.0), 3),
            stage_analysis_seconds=round(self._stage_times.get("analysis", 0.0), 3),
            stage_advanced_metrics_seconds=round(self._stage_times.get("advanced_metrics", 0.0), 3),
            stage_save_seconds=round(self._stage_times.get("save", 0.0), 3),
            peak_memory_mb=round(self._peak_memory_mb, 1),
            peak_gpu_memory_mb=round(self._peak_gpu_memory_mb, 1),
            gpu_utilization_pct=getattr(self, "_gpu_utilization", 0.0),
            gpu_name=self._system_info["gpu_name"],
            cpu_name=self._system_info["cpu_name"],
            ram_gb=self._system_info["ram_gb"],
            model_size=model_size,
            frame_skip=frame_skip,
        )
        logger.info(
            f"Benchmark: {result.total_time_seconds:.1f}s for {video_duration_seconds:.1f}s video "
            f"(ratio={result.realtime_ratio:.2f}, fps={result.fps_effective:.1f})"
        )
        return result

    def reset(self) -> None:
        """Reset all timers for a new benchmark run."""
        self._stage_times = {}
        self._start_times = {}
        self._peak_memory_mb = 0.0
        self._peak_gpu_memory_mb = 0.0
        self._gpu_utilization = 0.0

    def get_baseline_summary(self, storage_service) -> dict[str, Any]:
        """Get average performance from recent benchmarks."""
        try:
            rows = storage_service._conn.execute(  # type: ignore[union-attr]
                """
                SELECT
                    AVG(total_time_seconds) as avg_time,
                    AVG(realtime_ratio) as avg_ratio,
                    AVG(fps_effective) as avg_fps,
                    AVG(peak_memory_mb) as avg_mem,
                    AVG(peak_gpu_memory_mb) as avg_gpu_mem,
                    COUNT(*) as count
                FROM benchmark_results
                WHERE created_at > datetime('now', '-7 days')
                """
            ).fetchone()
            return {
                "avg_time_seconds": round(rows[0] or 0, 1),
                "avg_realtime_ratio": round(rows[1] or 0, 2),
                "avg_fps_effective": round(rows[2] or 0, 1),
                "avg_peak_memory_mb": round(rows[3] or 0, 1),
                "avg_peak_gpu_memory_mb": round(rows[4] or 0, 1),
                "sample_count": rows[5] or 0,
            }
        except Exception as e:
            logger.warning(f"Failed to get baseline: {e}")
            return {}

    @staticmethod
    def classify_gpu_tier(gpu_name: str) -> str:
        """Classify GPU into performance tier for recommendations."""
        name_lower = gpu_name.lower()
        # High-end
        if any(x in name_lower for x in ["4090", "4080", "3090", "a100", "h100"]):
            return "high"
        # Mid-range
        if any(x in name_lower for x in ["4070", "4060", "3080", "3070", "3060", "2080"]):
            return "mid"
        # Entry / low
        if any(x in name_lower for x in ["4050", "3050", "2060", "1660", "1650"]):
            return "low"
        return "unknown"

    @staticmethod
    def recommend_settings(gpu_tier: str) -> dict[str, Any]:
        """Recommend analysis settings based on GPU tier."""
        tiers = {
            "high": {
                "model_size": "l",
                "frame_skip": 2,
                "gpu_enabled": True,
                "expected_ratio": 0.4,
            },
            "mid": {
                "model_size": "l",
                "frame_skip": 3,
                "gpu_enabled": True,
                "expected_ratio": 0.7,
            },
            "low": {
                "model_size": "m",
                "frame_skip": 4,
                "gpu_enabled": True,
                "expected_ratio": 1.2,
            },
            "unknown": {
                "model_size": "n",
                "frame_skip": 5,
                "gpu_enabled": False,
                "expected_ratio": 3.0,
            },
        }
        return tiers.get(gpu_tier, tiers["unknown"])

    @staticmethod
    async def measure_processing_speed(
        model_size: str,
        video_path: str | Path,
        gpu_enabled: bool = True,
        test_duration_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Measure processing speed for a given YOLO variant on a video clip.

        Processes the first N seconds of a video and returns FPS and realtime ratio.
        """
        import time as _time
        from pathlib import Path as _Path

        try:
            from kawkab.services.cv_service import CVService

            vp = _Path(video_path)
            if not vp.exists():
                return {"error": f"Video not found: {video_path}", "fps": 0.0}

            import cv2

            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                return {"error": "Cannot open video", "fps": 0.0}

            test_frames = int(test_duration_seconds * (cap.get(cv2.CAP_PROP_FPS) or 30.0))
            cap.release()

            svc = CVService(
                model_size=model_size,
                gpu_enabled=gpu_enabled,
                min_track_lifetime_frames=1,
            )
            await svc.initialize()

            start = _time.perf_counter()
            processed = 0
            cap = cv2.VideoCapture(str(vp))
            try:
                for _ in range(test_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    await svc.detect_frame(frame, processed, processed / 30.0)
                    processed += 1
            finally:
                cap.release()
                await svc.shutdown()

            elapsed = _time.perf_counter() - start
            fps = processed / elapsed if elapsed > 0 else 0.0
            realtime_ratio = elapsed / test_duration_seconds if test_duration_seconds > 0 else 0.0

            return {
                "variant": model_size,
                "frames_processed": processed,
                "elapsed_seconds": round(elapsed, 2),
                "fps": round(fps, 2),
                "realtime_ratio": round(realtime_ratio, 3),
            }
        except Exception as e:
            logger.warning(f"measure_processing_speed({model_size}) failed: {e}")
            return {"error": str(e), "fps": 0.0}

    @staticmethod
    async def auto_select_yolo_variant(
        video_path: str | Path,
        gpu_enabled: bool = True,
        min_realtime_ratio: float = 0.5,
        test_duration_seconds: float = 30.0,
    ) -> str:
        """Benchmark all YOLO variants and select the best one.

        Picks the largest variant that achieves >= min_realtime_ratio.
        Falls back to 'n' (nano) if none can reach the threshold.
        Results are cached by GPU model to avoid re-benchmarking.
        """
        from kawkab.core.gpu_acceleration import detect_gpu_tier as _tier

        gpu_tier = _tier()
        cache_key = f"gpu_{gpu_tier}"

        cache = _load_benchmark_cache()
        if cache_key in cache:
            cached = cache[cache_key]
            logger.info(
                f"Using cached benchmark for GPU tier={gpu_tier}: variant={cached['variant']}"
            )
            return cached["variant"]

        variants = ["n", "s", "m", "l", "x"]
        best_variant = "n"

        for var in variants:
            result = await BenchmarkService.measure_processing_speed(
                var,
                video_path,
                gpu_enabled=gpu_enabled,
                test_duration_seconds=min(test_duration_seconds, 15.0),
            )
            ratio = result.get("realtime_ratio", 999.0)
            logger.info(f"  yolo11{var}: {result.get('fps', 0):.1f} FPS (ratio={ratio:.2f})")
            if ratio <= min_realtime_ratio:
                best_variant = var
            else:
                break  # larger variants will only be slower

        cache[cache_key] = {"variant": best_variant, "gpu_tier": gpu_tier}
        _save_benchmark_cache(cache)
        logger.info(f"Auto-selected YOLO variant: yolo11{best_variant}")
        return best_variant
