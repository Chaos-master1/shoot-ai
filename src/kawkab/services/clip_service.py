"""Video clip extraction service.

Extracts video clips from match videos based on timestamps.
Uses FFmpeg for fast, lossless extraction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class ClipExtractionService:
    """Extracts video clips for evidence in coach reports."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        pre_pad_seconds: float = 2.0,
        post_pad_seconds: float = 2.0,
    ) -> None:
        self.cache_dir = cache_dir or Path.home() / "Documents" / "KawkabAI" / "clips"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pre_pad = pre_pad_seconds
        self.post_pad = post_pad_seconds
        logger.info(f"ClipExtractionService: cache={self.cache_dir}")

    async def extract_clip(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        output_name: str | None = None,
        quality: str = "medium",
    ) -> Path | None:
        """Extract a video clip between start_time and end_time.

        Args:
            video_path: Source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_name: Optional output filename (auto-generated if None)
            quality: "high" (libx264 crf 18), "medium" (crf 23), or "low" (crf 28)

        Returns:
            Path to extracted clip, or None if extraction failed
        """
        if not video_path.exists():
            logger.error(f"Source video not found: {video_path}")
            return None

        start_time = max(0.0, start_time - self.pre_pad)
        end_time = end_time + self.post_pad
        duration = end_time - start_time

        if duration <= 0:
            logger.warning(f"Invalid clip duration: {duration}s")
            return None

        if output_name is None:
            stem = video_path.stem
            output_name = f"{stem}_clip_{start_time:.0f}s_{end_time:.0f}s.mp4"
        output_path = self.cache_dir / output_name

        crf = {"high": 18, "medium": 23, "low": 28}.get(quality, 23)
        preset = {"high": "slow", "medium": "medium", "low": "ultrafast"}.get(quality, "medium")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.2f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.2f}",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error(f"FFmpeg failed: {stderr.decode()[:500]}")
                return None
            logger.info(f"Extracted clip: {output_path.name} ({duration:.1f}s)")
            return output_path
        except FileNotFoundError:
            logger.error("FFmpeg not found in PATH")
            return None
        except Exception as e:
            logger.error(f"Clip extraction failed: {e}")
            return None

    async def extract_evidence_clips(
        self,
        video_path: Path,
        timestamps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract multiple evidence clips and return their metadata.

        Args:
            video_path: Source video
            timestamps: List of dicts with 'start', 'end', 'description' keys

        Returns:
            List of dicts with 'start', 'end', 'description', 'path' keys
        """
        results = []
        for i, ts in enumerate(timestamps):
            start = ts.get("start", 0)
            end = ts.get("end", start + 5)
            desc = ts.get("description", f"Evidence clip {i + 1}")

            clip_path = await self.extract_clip(
                video_path=video_path,
                start_time=start,
                end_time=end,
                output_name=f"evidence_{i + 1}_{start:.0f}s.mp4",
                quality="medium",
            )
            if clip_path:
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "description": desc,
                        "path": str(clip_path),
                        "filename": clip_path.name,
                    }
                )
        return results

    async def extract_event_clips(
        self,
        video_path: Path,
        events: list[dict],
        context_seconds: float = 3.0,
    ) -> list[dict]:
        """Extract clips for detected events (passes, shots, tackles).

        Args:
            video_path: Source video
            events: List of detected events with 'timestamp' and 'type'
            context_seconds: Seconds of context before/after event

        Returns:
            List of event clips with metadata
        """
        results = []
        for i, event in enumerate(events[:50]):
            ts = event.get("timestamp", 0)
            event_type = event.get("type", "event")
            team = event.get("team", "unknown")
            clip_path = await self.extract_clip(
                video_path=video_path,
                start_time=ts - context_seconds,
                end_time=ts + context_seconds,
                output_name=f"event_{i + 1}_{event_type}_{ts:.0f}s.mp4",
                quality="medium",
            )
            if clip_path:
                results.append(
                    {
                        "event_index": i,
                        "event_type": event_type,
                        "team": team,
                        "timestamp": ts,
                        "clip_path": str(clip_path),
                        "filename": clip_path.name,
                    }
                )
        return results
