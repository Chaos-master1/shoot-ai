"""Tests for GPU acceleration detection and YOLO variant recommendation.

Covers: detect_gpu, detect_gpu_tier, recommend_yolo_variant, get_ffmpeg_gpu_args,
get_opencv_gpu_backend, is_hardware_decoding_available, optimize_opencv.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kawkab.core.gpu_acceleration import (
    detect_gpu,
    detect_gpu_tier,
    get_ffmpeg_gpu_args,
    get_opencv_gpu_backend,
    is_hardware_decoding_available,
    optimize_opencv,
    recommend_yolo_variant,
)


class TestDetectGpu:
    def test_cpu_fallback(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("importlib.import_module", side_effect=ImportError),
        ):
            assert detect_gpu() in ("cpu", "opencl")


class TestDetectGpuTier:
    def test_cpu_tier_when_no_gpu(self):
        with patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cpu"):
            assert detect_gpu_tier() == "cpu"

    def test_ultra_tier(self):
        with (
            patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cuda"),
            patch("subprocess.run") as mock_run,
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout.strip.return_value = "24576"
            mock_run.return_value = mock_result
            mock_result.stdout.strip = lambda: "24576"
            mock_result.stdout = "24576\n"
            assert detect_gpu_tier() == "ultra"

    def test_high_tier(self):
        with (
            patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cuda"),
            patch("subprocess.run") as mock_run,
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "16384\n"
            mock_run.return_value = mock_result
            assert detect_gpu_tier() == "high"

    def test_medium_tier(self):
        with (
            patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cuda"),
            patch("subprocess.run") as mock_run,
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "8192\n"
            mock_run.return_value = mock_result
            assert detect_gpu_tier() == "medium"

    def test_low_tier(self):
        with (
            patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cuda"),
            patch("subprocess.run") as mock_run,
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "4096\n"
            mock_run.return_value = mock_result
            assert detect_gpu_tier() == "low"

    def test_apple_mps_high(self):
        with (
            patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="mps"),
            patch("subprocess.run") as mock_run,
        ):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "    Apple M2 Pro\n"
            mock_run.return_value = mock_result
            assert detect_gpu_tier() in ("high", "medium")

    def test_nvidia_smi_failure_falls_to_medium(self):
        with (
            patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cuda"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            assert detect_gpu_tier() == "medium"


class TestRecommendYoloVariant:
    def test_ultra_yields_x(self):
        assert recommend_yolo_variant("ultra") == "x"

    def test_high_yields_l(self):
        assert recommend_yolo_variant("high") == "l"

    def test_medium_yields_m(self):
        assert recommend_yolo_variant("medium") == "m"

    def test_low_yields_s(self):
        assert recommend_yolo_variant("low") == "s"

    def test_cpu_yields_n(self):
        assert recommend_yolo_variant("cpu") == "n"

    def test_auto_detect(self):
        with patch("kawkab.core.gpu_acceleration.detect_gpu_tier", return_value="high"):
            assert recommend_yolo_variant() == "l"

    def test_unknown_tier_defaults_to_l(self):
        assert recommend_yolo_variant("unknown") == "l"  # type: ignore


class TestGetFfmpegGpuArgs:
    @patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cuda")
    def test_cuda_args(self, _mock):
        args = get_ffmpeg_gpu_args()
        assert "-hwaccel" in args
        assert "cuda" in args

    @patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="mps")
    def test_mps_args(self, _mock):
        args = get_ffmpeg_gpu_args()
        assert "videotoolbox" in args

    @patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cpu")
    def test_cpu_args(self, _mock):
        assert get_ffmpeg_gpu_args() == []


class TestGetOpencvGpuBackend:
    @patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cpu")
    def test_cpu_returns_none(self, _mock):
        assert get_opencv_gpu_backend() is None


class TestIsHardwareDecodingAvailable:
    @patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cuda")
    def test_cuda_available(self, _mock):
        assert is_hardware_decoding_available() is True

    @patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cpu")
    def test_cpu_not_available(self, _mock):
        assert is_hardware_decoding_available() is False


class TestOptimizeOpencv:
    @patch("kawkab.core.gpu_acceleration.detect_gpu", return_value="cpu")
    def test_returns_dict(self, _mock):
        result = optimize_opencv()
        assert isinstance(result, dict)
        assert "gpu" in result
        assert "opencl" in result
        assert "ipp" in result
