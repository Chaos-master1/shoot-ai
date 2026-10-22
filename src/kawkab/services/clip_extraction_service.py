"""Video clip library service for coaching presentations (v0.8.2+).

Manages a library of extracted video clips with thumbnails, playlists, and
DB-backed storage integration. Used for:
- Coaching presentations
- Player review sessions
- Scout reports
- Social media sharing

.. note::
    This service is the **library** layer. The lower-level fire-and-forget
    clip extraction lives in ``kawkab.services.clip_service.ClipExtractionService``
    and is wired into the production pipeline via ``app.py`` and ``ui/bridge.py``.

    This module was renamed in cycle F of ITERATION_LOG.md to avoid a class
    name collision with ``clip_service.py``. The old class name
    ``ClipExtractionService`` is now ``ClipLibraryService``.

Features:
- Extract clips by event type (goals, shots, tackles, etc.)
- Custom padding before/after event
- Export to MP4 with overlays
- Organize clips into playlists
- Thumbnail generation
- DB-backed storage integration (via StorageService)
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


@dataclass
class VideoClip:
    """A extracted video clip with metadata."""

    clip_id: int
    match_id: int
    event_type: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    source_video_path: str
    output_path: str
    thumbnail_path: str | None = None
    player_id: int | None = None
    description: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClipPlaylist:
    """A collection of clips for a presentation."""

    playlist_id: int
    name: str
    description: str
    clip_ids: list[int]
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClipLibraryService:
    """Extract and manage video clips for coaching presentations.

    .. deprecated-rename::
        Previously named ``ClipExtractionService``. Renamed in cycle F of
        ITERATION_LOG.md to disambiguate from the lower-level
        ``clip_service.ClipExtractionService`` that is wired into the
        production pipeline.
    """

    # Default padding around events (seconds)
    DEFAULT_PADDING = 5.0

    def __init__(self, storage_service=None, output_dir: Path | None = None) -> None:
        self.storage = storage_service
        self.output_dir = output_dir or (get_paths().exports / "clips")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_clip_path(self, match_id: int, clip_id: int, event_type: str) -> Path:
        """Generate a unique path for a clip file."""
        match_dir = self.output_dir / f"match_{match_id}"
        match_dir.mkdir(exist_ok=True)
        return match_dir / f"{event_type}_{clip_id}.mp4"

    def _get_thumbnail_path(self, match_id: int, clip_id: int) -> Path:
        """Generate a unique path for a thumbnail."""
        match_dir = self.output_dir / f"match_{match_id}"
        match_dir.mkdir(exist_ok=True)
        return match_dir / f"thumb_{clip_id}.jpg"

    def extract_clip(
        self,
        source_video: str,
        start_seconds: float,
        end_seconds: float,
        output_path: str,
    ) -> bool:
        """Extract a clip from a video using ffmpeg.

        Args:
            source_video: Path to source video
            start_seconds: Start time in seconds
            end_seconds: End time in seconds
            output_path: Output clip path

        Returns:
            True if successful
        """
        duration = end_seconds - start_seconds
        if duration <= 0:
            logger.error(f"Invalid clip duration: {duration}")
            return False

        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-ss",
                str(start_seconds),
                "-t",
                str(duration),
                "-i",
                source_video,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return False

            logger.info(f"Clip extracted: {output_path} ({duration:.1f}s)")
            return True
        except FileNotFoundError:
            logger.error("ffmpeg not found. Install ffmpeg to extract clips.")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Clip extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Clip extraction failed: {e}")
            return False

    def generate_thumbnail(self, video_path: str, output_path: str, timestamp: float = 0.0) -> bool:
        """Generate a thumbnail from a video frame.

        Args:
            video_path: Path to video
            output_path: Output thumbnail path
            timestamp: Frame timestamp (default: first frame)

        Returns:
            True if successful
        """
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(timestamp),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-q:v",
                "2",
                output_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error(f"Thumbnail generation failed: {result.stderr}")
                return False
            return True
        except FileNotFoundError:
            logger.error("ffmpeg not found")
            return False
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            return False

    async def create_clip_from_event(
        self,
        match_id: int,
        event: dict,
        padding_seconds: float | None = None,
    ) -> VideoClip | None:
        """Create a clip from an event dictionary.

        Args:
            match_id: Match ID
            event: Event dict with 'type', 'timestamp', 'video_path'
            padding_seconds: Padding before/after event

        Returns:
            VideoClip or None
        """
        padding = padding_seconds or self.DEFAULT_PADDING
        event_type = event.get("type", "unknown")
        timestamp = event.get("timestamp", 0.0)
        video_path = event.get("video_path", "")
        player_id = event.get("player_id")

        if not video_path or not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            return None

        start_seconds = max(0.0, timestamp - padding)
        end_seconds = timestamp + padding

        # Get video duration to clamp end_seconds
        try:
            import ffmpeg

            probe = ffmpeg.probe(video_path)
            duration = float(probe["format"]["duration"])
            end_seconds = min(end_seconds, duration)
        except Exception:
            pass  # Use unclamped value

        clip_id = 1  # Will be replaced by DB ID
        output_path = self._get_clip_path(match_id, clip_id, event_type)
        thumbnail_path = self._get_thumbnail_path(match_id, clip_id)

        success = self.extract_clip(video_path, start_seconds, end_seconds, str(output_path))
        if not success:
            return None

        self.generate_thumbnail(str(output_path), str(thumbnail_path), padding)

        clip = VideoClip(
            clip_id=clip_id,
            match_id=match_id,
            event_type=event_type,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=end_seconds - start_seconds,
            source_video_path=video_path,
            output_path=str(output_path),
            thumbnail_path=str(thumbnail_path) if thumbnail_path.exists() else None,
            player_id=player_id,
            description=event.get("description", ""),
        )

        if self.storage is not None:
            clip_id = await self.storage.save_clip(clip.to_dict())
            clip.clip_id = clip_id

        return clip

    async def create_clips_from_events(
        self,
        match_id: int,
        events: list[dict],
        event_types: list[str] | None = None,
        padding_seconds: float | None = None,
    ) -> list[VideoClip]:
        """Create clips from a list of events.

        Args:
            match_id: Match ID
            events: List of event dictionaries
            event_types: Filter by event types (None = all)
            padding_seconds: Padding around each event

        Returns:
            List of successfully created VideoClips
        """
        clips = []
        for event in events:
            if event_types and event.get("type") not in event_types:
                continue
            clip = await self.create_clip_from_event(match_id, event, padding_seconds)
            if clip:
                clips.append(clip)
        return clips

    async def create_playlist(
        self, name: str, clip_ids: list[int], description: str = ""
    ) -> ClipPlaylist | None:
        """Create a playlist of clips.

        Args:
            name: Playlist name
            clip_ids: List of clip IDs
            description: Optional description

        Returns:
            ClipPlaylist or None
        """
        playlist = ClipPlaylist(
            playlist_id=0,
            name=name,
            description=description,
            clip_ids=clip_ids,
        )
        if self.storage is not None:
            playlist_id = await self.storage.save_playlist(playlist.to_dict())
            playlist.playlist_id = playlist_id
        return playlist

    def get_clip_directory(self, match_id: int) -> Path:
        """Get the directory containing clips for a match."""
        return self.output_dir / f"match_{match_id}"
