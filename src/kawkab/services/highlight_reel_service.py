from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass

from kawkab.core.logging import get_logger
from kawkab.core.security import SecurityValidator

logger = get_logger(__name__)


@dataclass
class ReelClip:
    video_path: str
    start_seconds: float
    end_seconds: float
    label: str = ""


@dataclass
class ReelResult:
    output_path: str
    clip_count: int
    total_duration_seconds: float
    method: str = "ffmpeg_concat"


class HighlightReelService:
    def __init__(self, output_dir: str | None = None):
        self._output_dir = output_dir or tempfile.gettempdir()

    async def extract_clip_segment(
        self, video_path: str, start: float, end: float, output_path: str
    ) -> bool:
        duration = end - start
        if duration <= 0:
            return False
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            video_path,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0 and os.path.isfile(output_path)
        except Exception as e:
            logger.error(f"extract_clip_segment failed: {e}")
            return False

    async def compose_reel(
        self, clips: list[ReelClip], output_filename: str = "highlight_reel.mp4"
    ) -> str:
        if not clips:
            return json.dumps({"error": "No clips provided"})
        validated_clips = []
        for c in clips:
            path = SecurityValidator.validate_video_path(c.video_path)
            if path and os.path.isfile(path) and c.end_seconds > c.start_seconds:
                validated_clips.append(c)
        if not validated_clips:
            return json.dumps({"error": "No valid clips"})
        segment_dir = tempfile.mkdtemp(prefix="reel_")
        segment_paths = []
        try:
            tasks = []
            for i, c in enumerate(validated_clips):
                seg_path = os.path.join(segment_dir, f"seg_{i:04d}.mp4")
                tasks.append(
                    self.extract_clip_segment(
                        c.video_path, c.start_seconds, c.end_seconds, seg_path
                    )
                )
                segment_paths.append((seg_path, c))
            results = await asyncio.gather(*tasks)
            valid_segments = [
                (p, c) for (p, c), ok in zip(segment_paths, results, strict=False) if ok
            ]
            if not valid_segments:
                return json.dumps({"error": "No segments could be extracted"})
            concat_path = os.path.join(segment_dir, "concat_list.txt")
            with open(concat_path, "w") as f:
                for seg_path, _ in valid_segments:
                    f.write(f"file '{seg_path}'\n")
            output_path = os.path.join(self._output_dir, output_filename)
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "22",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                output_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *concat_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0 or not os.path.isfile(output_path):
                return json.dumps({"error": "FFmpeg concat failed"})
            total_duration = sum(c.end_seconds - c.start_seconds for _, c in valid_segments)
            return json.dumps(
                {
                    "output_path": output_path,
                    "clip_count": len(valid_segments),
                    "total_duration_s": round(total_duration, 1),
                }
            )
        except Exception as e:
            logger.error(f"compose_reel failed: {e}")
            return json.dumps({"error": str(e)})
        finally:
            for seg_path, _ in segment_paths:
                try:
                    if os.path.isfile(seg_path):
                        os.remove(seg_path)
                except OSError:
                    pass
            try:
                if os.path.isfile(concat_path):
                    os.remove(concat_path)
                os.rmdir(segment_dir)
            except OSError:
                pass

    def make_reel_from_events(
        self,
        match_id: int,
        events: list[dict],
        video_path: str,
        context_seconds: float = 3.0,
        output_filename: str = "event_reel.mp4",
    ) -> str:
        clips = []
        for e in events:
            ts = e.get("timestamp", 0)
            start = max(0.0, ts - context_seconds)
            end = ts + context_seconds
            clips.append(
                ReelClip(
                    video_path=video_path,
                    start_seconds=start,
                    end_seconds=end,
                    label=e.get("type", "event"),
                )
            )
        import json as _json

        return _json.dumps({"clip_count": len(clips), "clips_defined": True})
