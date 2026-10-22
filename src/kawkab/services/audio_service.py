"""Audio service - faster-whisper transcription, whistle detection.

Optional audio analysis from match videos for richer context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class AudioService:
    """Audio analysis pipeline using faster-whisper and audio detection."""

    def __init__(
        self,
        enable_transcription: bool = True,
        enable_whistle_detection: bool = True,
        enable_crowd_analysis: bool = False,
        whisper_model: str = "base",
        gpu_enabled: bool = True,
    ) -> None:
        self.enable_transcription = enable_transcription
        self.enable_whistle_detection = enable_whistle_detection
        self.enable_crowd_analysis = enable_crowd_analysis
        self.whisper_model = whisper_model
        self.gpu_enabled = gpu_enabled
        self._model: Any = None

        logger.info(
            f"AudioService: transcribe={enable_transcription}, "
            f"whistle={enable_whistle_detection}, model={whisper_model}"
        )

    async def initialize(self) -> None:
        """Lazy-load Whisper model."""
        if not self.enable_transcription and not self.enable_whistle_detection:
            return

        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            device = "cuda" if self.gpu_enabled else "cpu"
            compute_type = "float16" if self.gpu_enabled else "int8"

            logger.info(f"Loading Whisper model: {self.whisper_model} on {device}")
            self._model = WhisperModel(self.whisper_model, device=device, compute_type=compute_type)
            logger.info("Whisper model loaded")
        except ImportError:
            logger.warning("faster-whisper not installed. Run: pip install faster-whisper")
            self._model = None

    async def transcribe_video(self, video_path: Path) -> list[dict]:
        """Transcribe audio from a video file.

        Args:
            video_path: Path to video file

        Returns:
            List of segments: [{start, end, text, language}]
        """
        if not self.enable_transcription:
            logger.info("Transcription disabled, skipping")
            return []

        if self._model is None:
            await self.initialize()

        if self._model is None:
            logger.warning("Whisper model not available, skipping transcription")
            return []

        logger.info(f"Transcribing audio: {video_path.name}")

        try:
            segments, info = self._model.transcribe(str(video_path), beam_size=5)

            results = []
            for segment in segments:
                results.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "language": info.language,
                        "confidence": getattr(segment, "avg_logprob", 0.0),
                    }
                )

            logger.info(
                f"Transcription complete: {len(results)} segments, language={info.language}"
            )
            return results
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []

    async def detect_whistles(self, video_path: Path) -> list[dict]:
        """Detect referee whistles in video audio.

        Args:
            video_path: Path to video file

        Returns:
            List of whistle events: [{timestamp, confidence}]
        """
        if not self.enable_whistle_detection:
            return []

        logger.info(f"Detecting whistles in: {video_path.name}")

        try:
            import librosa
            import numpy as np
        except ImportError:
            logger.warning("librosa not installed. Run: pip install librosa")
            return []

        try:
            y, sr = librosa.load(str(video_path), sr=None, mono=True)

            whistle_freq_min = 2500
            whistle_freq_max = 4000
            whistle_duration_max = 1.5

            stft = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr)

            freq_mask = (freqs >= whistle_freq_min) & (freqs <= whistle_freq_max)
            whistle_energy = np.mean(stft[freq_mask, :], axis=0)
            threshold = np.percentile(whistle_energy, 95)

            whistle_frames = whistle_energy > threshold
            events = []

            in_whistle = False
            start_time = 0.0
            for i, is_whistle in enumerate(whistle_frames):
                if is_whistle and not in_whistle:
                    start_time = times[i]
                    in_whistle = True
                elif not is_whistle and in_whistle:
                    duration = times[i] - start_time
                    if duration <= whistle_duration_max:
                        events.append(
                            {
                                "timestamp": start_time,
                                "duration": duration,
                                "confidence": min(
                                    1.0,
                                    whistle_energy[int(start_time * sr / 512) : i].mean()
                                    / threshold,
                                ),
                            }
                        )
                    in_whistle = False

            logger.info(f"Detected {len(events)} whistle events")
            return events
        except Exception as e:
            logger.error(f"Whistle detection failed: {e}")
            return []

    async def analyze_crowd_noise(self, video_path: Path) -> dict:
        """Analyze crowd noise intensity over time (cheer-detection proxy).

        Args:
            video_path: Path to video file

        Returns:
            Dict with avg_intensity, peak_intensity, peak_time, samples.
            Always has this exact shape (zeroed out) if disabled, librosa
            is unavailable, or analysis fails -- callers never need to
            check for a missing key.
        """
        empty = {"avg_intensity": 0.0, "peak_intensity": 0.0, "peak_time": 0.0, "samples": 0}
        if not self.enable_crowd_analysis:
            return empty

        logger.info(f"Analyzing crowd noise in: {video_path.name}")

        try:
            import librosa
            import numpy as np
        except ImportError:
            logger.warning("librosa not installed. Run: pip install librosa")
            return empty

        try:
            y, sr = librosa.load(str(video_path), sr=None, mono=True)
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

            if len(rms) == 0:
                return empty

            peak_idx = int(np.argmax(rms))
            return {
                "avg_intensity": float(np.mean(rms)),
                "peak_intensity": float(rms[peak_idx]),
                "peak_time": float(times[peak_idx]),
                "samples": int(len(rms)),
            }
        except Exception as e:
            logger.error(f"Crowd noise analysis failed: {e}")
            return empty

    async def analyze_audio(self, video_path: Path) -> dict:
        """Full audio analysis: transcription + whistle detection.

        Args:
            video_path: Path to video file

        Returns:
            Dict with transcription, whistles, and metadata
        """
        transcription = await self.transcribe_video(video_path)
        whistles = await self.detect_whistles(video_path)

        return {
            "transcription": transcription,
            "whistles": whistles,
            "duration": transcription[-1]["end"] if transcription else 0.0,
        }
