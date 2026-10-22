"""Video enhancement service - FFmpeg filters, Real-ESRGAN, RIFE.

Preprocesses amateur phone footage: stabilization, denoising, upscaling,
frame interpolation. All runs locally on GPU.
"""

from __future__ import annotations

from pathlib import Path

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class EnhancementService:
    """Video enhancement pipeline using FFmpeg, Real-ESRGAN, and RIFE."""

    def __init__(
        self,
        enable_stabilization: bool = True,
        enable_denoising: bool = True,
        enable_sharpening: bool = True,
        enable_upscaling: bool = False,
        enable_interpolation: bool = False,
        gpu_enabled: bool = True,
    ) -> None:
        self.enable_stabilization = enable_stabilization
        self.enable_denoising = enable_denoising
        self.enable_sharpening = enable_sharpening
        self.enable_upscaling = enable_upscaling
        self.enable_interpolation = enable_interpolation
        self.gpu_enabled = gpu_enabled

        logger.info(
            f"EnhancementService: stab={enable_stabilization}, "
            f"denoise={enable_denoising}, sharp={enable_sharpening}, "
            f"upscale={enable_upscaling}, interp={enable_interpolation}"
        )

    async def preprocess_video(self, input_path: Path, output_path: Path) -> Path:
        """Apply free FFmpeg filters to improve amateur footage.

        Filters applied (in order):
        - vidstabdetect + vidstabtransform: camera stabilization
        - hqdn3d: noise reduction
        - unsharp: mild sharpening
        - scale: resolution normalization to 720p

        Args:
            input_path: Source video file
            output_path: Where to save preprocessed video

        Returns:
            Path to preprocessed video
        """
        import ffmpeg

        logger.info(f"Preprocessing video: {input_path.name}")

        stream = ffmpeg.input(str(input_path))

        if self.enable_stabilization:
            stream = stream.filter("vidstabdetect", shakiness=8, accuracy=15)
            stream = stream.filter("vidstabtransform", input=str(input_path), smoothing=20)

        if self.enable_denoising:
            stream = stream.filter("hqdn3d", luma_spatial=4, chroma_spatial=3)

        if self.enable_sharpening:
            stream = stream.filter("unsharp", luma_msize_x=5, luma_msize_y=5, luma_amount=1.0)

        stream = stream.filter("scale", 1280, 720)
        stream = stream.output(
            str(output_path), vcodec="libx264", preset="fast", crf=23
        ).overwrite_output()

        stream.run(quiet=True)
        logger.info(f"Preprocessed video saved: {output_path.name}")
        return output_path

    async def upscale_video(self, input_path: Path, output_path: Path, scale: int = 2) -> Path:
        """Upscale video using Real-ESRGAN (optional, GPU-intensive).

        Args:
            input_path: Source video
            output_path: Where to save upscaled video
            scale: Upscale factor (2 or 4)

        Returns:
            Path to upscaled video
        """
        if not self.enable_upscaling:
            logger.info("Upscaling disabled, skipping")
            return input_path

        logger.info(f"Upscaling video: {input_path.name} (scale={scale}x) via Real-ESRGAN")

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            logger.error("Real-ESRGAN not installed. Run: pip install realesrgan basicsr")
            return input_path

        import cv2
        import torch

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

        upsampler = RealESRGANer(
            scale=scale,
            model_path="weights/RealESRGAN_x4plus.pth",
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=self.gpu_enabled and torch.cuda.is_available(),
        )

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 runtime attr absent from stale stubs
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w * scale, h * scale))

        frame_num = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                upscaled, _ = upsampler.enhance(frame, outscale=scale)
                out.write(upscaled)
                frame_num += 1

                if frame_num % 30 == 0:
                    logger.debug(f"Upscaled {frame_num}/{total} frames")
        finally:
            cap.release()
            out.release()

        logger.info(f"Upscaled video saved: {output_path.name}")
        return output_path

    async def interpolate_video(
        self, input_path: Path, output_path: Path, target_fps: int = 50
    ) -> Path:
        """Interpolate frames using RIFE (optional, GPU-intensive).

        Args:
            input_path: Source video
            output_path: Where to save interpolated video
            target_fps: Target frame rate (e.g., 50 for 2x interpolation)

        Returns:
            Path to interpolated video
        """
        if not self.enable_interpolation:
            logger.info("Interpolation disabled, skipping")
            return input_path

        logger.info(f"Interpolating video: {input_path.name} → {target_fps} FPS via RIFE")

        try:
            from rife.RIFE_HDv3 import Model
        except ImportError:
            logger.error(
                "RIFE not installed. See: https://github.com/megvii-research/ECCV2022-RIFE"
            )
            return input_path

        import cv2
        import numpy as np
        import torch

        model = Model()
        model.load_model("rife")
        model.eval()
        model.device()

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ratio = target_fps / fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 runtime attr absent from stale stubs
        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return input_path

        try:
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break

                out.write(prev_frame)

                n_intermediate = int(ratio) - 1
                for i in range(1, n_intermediate + 1):
                    t = i / ratio
                    middle = model.inference(
                        torch.from_numpy(prev_frame).float(),
                        torch.from_numpy(curr_frame).float(),
                        t,
                    )
                    middle_np = middle.numpy().astype(np.uint8)
                    out.write(middle_np)

                prev_frame = curr_frame
        finally:
            cap.release()
            out.release()

        logger.info(f"Interpolated video saved: {output_path.name}")
        return output_path
