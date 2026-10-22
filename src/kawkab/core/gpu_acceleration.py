"""GPU acceleration detection and pipeline integration for video processing.

Supports CUDA (NVIDIA), Metal (Apple MPS), OpenCL fallback.
Auto-detects best available backend and configures OpenCV + FFmpeg accordingly.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Literal

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

GPUBackend = Literal["cuda", "mps", "opencl", "cpu"]
GPUTier = Literal["ultra", "high", "medium", "low", "cpu"]


def detect_gpu() -> GPUBackend:
    """Detect the best available GPU backend for video processing."""
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split("\n")[0]
                logger.info(f"CUDA GPU detected: {gpu_name}")
                return "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    elif system == "Darwin":
        try:
            import ctypes

            lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
            if lib:
                logger.info("Apple Metal (MPS) backend detected")
                return "mps"
        except Exception:
            pass

    try:
        import cv2

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("OpenCV CUDA enabled device found")
            return "cuda"
    except (ImportError, AttributeError):
        pass

    try:
        import cv2

        if hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL():
            logger.info("OpenCL backend available via OpenCV")
            return "opencl"
    except ImportError:
        pass

    logger.info("No GPU backend detected — using CPU")
    return "cpu"


_FFMPEG_GPU_ARGS: dict[str, list[str]] = {
    "cuda": [
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-extra_hw_frames",
        "8",
    ],
    "mps": ["-hwaccel", "videotoolbox"],
    "opencl": ["-hwaccel", "opencl"],
}


def get_ffmpeg_gpu_args() -> list[str]:
    """Get FFmpeg GPU acceleration arguments for the detected backend."""
    return _FFMPEG_GPU_ARGS.get(detect_gpu(), [])


def get_opencv_gpu_backend() -> int | None:
    """Get the OpenCV video backend constant for the detected GPU."""
    backend = detect_gpu()
    try:
        import cv2

        if backend == "cuda":
            return cv2.CAP_FFMPEG
        elif backend == "mps":
            if hasattr(cv2, "CAP_AVFOUNDATION"):
                return cv2.CAP_AVFOUNDATION
        elif backend == "opencl" and hasattr(cv2, "CAP_OPENCV_MJPEG"):
            return cv2.CAP_OPENCV_MJPEG
    except ImportError:
        pass
    return None


def is_hardware_decoding_available() -> bool:
    """Check if hardware-accelerated video decoding is available."""
    backend = detect_gpu()
    return backend != "cpu"


def optimize_opencv() -> dict[str, bool]:
    """Configure OpenCV to use optimal settings for the detected hardware."""
    results = {"gpu": False, "opencl": False, "ipp": False}

    try:
        import cv2

        backend = detect_gpu()
        if backend == "cuda":
            cv2.setUseOptimized(True)
            results["gpu"] = True
            logger.info("OpenCV GPU optimizations enabled")
        elif backend == "opencl":
            if hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                results["opencl"] = True
                logger.info("OpenCV OpenCL optimizations enabled")
        else:
            if cv2.useOptimized():
                results["ipp"] = True
                logger.info("OpenCV IPP optimizations enabled")
    except ImportError:
        pass

    # Set FFmpeg thread count based on CPU cores
    try:
        cpu_count = os.cpu_count() or 4
        thread_count = max(2, min(cpu_count, 16))
        os.environ.setdefault("FFMPEG_THREADS", str(thread_count))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(thread_count))
        os.environ.setdefault("MKL_NUM_THREADS", str(thread_count))
        logger.info(f"Thread count set to {thread_count}")
    except Exception:
        pass

    return results


def detect_gpu_tier() -> GPUTier:
    """Detect GPU tier based on VRAM.

    Tiers:
      - 'ultra': 24GB+ VRAM (RTX 4090, A100, etc.)
      - 'high':  12–23 GB (RTX 4080, RTX 3080, etc.)
      - 'medium': 6–11 GB (RTX 3060, GTX 1660, etc.)
      - 'low':    <6 GB or integrated
      - 'cpu':    no GPU detected
    """
    import subprocess as _sub

    backend = detect_gpu()
    if backend == "cpu":
        return "cpu"

    if backend == "mps":
        try:
            result = _sub.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "Apple M" in result.stdout:
                return "high"
        except Exception:
            pass
        return "medium"

    # NVIDIA: query VRAM via nvidia-smi
    try:
        result = _sub.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            mem_mb = int(result.stdout.strip().split("\n")[0])
            if mem_mb >= 24000:
                return "ultra"
            elif mem_mb >= 12000:
                return "high"
            elif mem_mb >= 6000:
                return "medium"
            else:
                return "low"
    except (FileNotFoundError, _sub.TimeoutExpired, ValueError, IndexError):
        pass

    return "medium"


def recommend_yolo_variant(tier: GPUTier | None = None) -> str:
    """Recommend YOLO model size based on GPU tier.

    Returns one of 'n', 's', 'm', 'l', 'x'.
    Caller passes 'auto' as model_size to trigger this.
    """
    if tier is None:
        tier = detect_gpu_tier()
    mapping: dict[GPUTier, str] = {
        "ultra": "x",
        "high": "l",
        "medium": "m",
        "low": "s",
        "cpu": "n",
    }
    variant = mapping.get(tier, "l")
    logger.info(f"GPU tier={tier} → recommended YOLO variant=yolo11{variant}")
    return variant


__all__ = [
    "detect_gpu",
    "get_ffmpeg_gpu_args",
    "get_opencv_gpu_backend",
    "is_hardware_decoding_available",
    "optimize_opencv",
    "detect_gpu_tier",
    "recommend_yolo_variant",
]
