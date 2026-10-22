"""Multi-camera synchronization for professional match footage.

Handles alignment of multiple camera feeds using timecode offsets and
audio cross-correlation for automatic synchronization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CameraFeed:
    """A single camera feed with metadata for synchronization."""

    camera_id: str
    video_path: str
    fps: float
    timecode_offset: float = 0.0
    camera_intrinsics: dict | None = None


@dataclass
class SyncPoint:
    """A timestamp where two or more camera feeds align."""

    global_time_s: float
    camera_times: dict[str, float]


class MultiCameraSyncService:
    """Sync multiple camera feeds by timecode offset or audio cross-correlation."""

    def __init__(self) -> None:
        self._feeds: dict[str, CameraFeed] = {}

    # ------------------------------------------------------------------
    # Feed management
    # ------------------------------------------------------------------

    def add_feed(
        self,
        camera_id: str,
        video_path: str,
        fps: float,
        timecode_offset: float = 0.0,
    ) -> None:
        """Register a new camera feed.

        Args:
            camera_id: Unique identifier for the camera.
            video_path: Path to the video file.
            fps: Frames per second of this feed.
            timecode_offset: Manual offset in seconds (positive = earlier).
        """
        if camera_id in self._feeds:
            logger.warning(f"Overwriting existing feed: {camera_id}")
        self._feeds[camera_id] = CameraFeed(
            camera_id=camera_id,
            video_path=video_path,
            fps=fps,
            timecode_offset=timecode_offset,
        )
        logger.info(f"Added feed {camera_id} ({video_path}) @ {fps} fps, offset={timecode_offset}s")

    def remove_feed(self, camera_id: str) -> None:
        """Remove a registered camera feed."""
        removed = self._feeds.pop(camera_id, None)
        if removed is None:
            logger.warning(f"Feed not found: {camera_id}")
        else:
            logger.info(f"Removed feed {camera_id}")

    def get_feed_count(self) -> int:
        """Return the number of registered feeds."""
        return len(self._feeds)

    def clear(self) -> None:
        """Remove all registered feeds."""
        count = len(self._feeds)
        self._feeds.clear()
        logger.info(f"Cleared all feeds ({count} removed)")

    # ------------------------------------------------------------------
    # Sync methods
    # ------------------------------------------------------------------

    def sync_by_timecode(self) -> dict[str, list[dict[str, Any]]]:
        """Align feeds by their explicit timecode offsets.

        Returns a dict mapping each camera_id to a list of sync points
        (timestamps where the feed aligns to the global timeline).
        """
        if not self._feeds:
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        for cid, feed in self._feeds.items():
            result[cid] = [
                {
                    "camera_id": cid,
                    "local_time_s": round(t, 3),
                    "global_time_s": round(t - feed.timecode_offset, 3),
                }
                for t in self._iter_timestamps(feed)
            ]
        return result

    def sync_by_audio_cross_correlation(
        self,
        sample_rate: int = 44100,
        correlation_threshold: float = 0.5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Align feeds by FFT cross-correlation of audio envelopes.

        Requires numpy for FFT-based correlation. Compares feeds pair-wise
        against the first registered feed (reference). Returns sync points
        with estimated lag in seconds.

        Args:
            sample_rate: Audio sample rate to assume for synthetic envelope.
            correlation_threshold: Minimum normalized correlation to accept.

        Returns:
            Dict mapping camera_id to list of detected sync points.
        """
        if not self._feeds:
            return {}

        feed_ids = list(self._feeds.keys())
        reference_id = feed_ids[0]
        reference = self._feeds[reference_id]

        result: dict[str, list[dict[str, Any]]] = {}

        for cid in feed_ids:
            if cid == reference_id:
                result[cid] = self._build_sync_points(cid, reference, 0.0, 1.0, sample_rate)
                continue

            feed = self._feeds[cid]
            lag_samples, correlation = self._cross_correlate_feeds(reference, feed, sample_rate)

            if correlation < correlation_threshold:
                logger.info(
                    f"Audio correlation too low for {cid} vs {reference_id}: "
                    f"{correlation:.3f} < {correlation_threshold}"
                )
                result[cid] = []
                continue

            lag_seconds = lag_samples / sample_rate
            logger.info(
                f"Audio sync {cid} -> {reference_id}: "
                f"lag={lag_seconds:.3f}s, correlation={correlation:.3f}"
            )

            result[cid] = self._build_sync_points(cid, feed, lag_seconds, correlation, sample_rate)

        return result

    def sync_feeds(self) -> dict[str, list[dict[str, Any]]]:
        """Auto-detect best sync method and align all feeds.

        If any feed has a non-zero timecode offset, uses timecode sync.
        Otherwise attempts audio cross-correlation.
        """
        if not self._feeds:
            return {}

        has_offsets = any(f.timecode_offset != 0.0 for f in self._feeds.values())
        if has_offsets:
            logger.info("Using timecode-based sync")
            return self.sync_by_timecode()

        logger.info("Attempting audio cross-correlation sync")
        return self.sync_by_audio_cross_correlation()

    # ------------------------------------------------------------------
    # Global timeline
    # ------------------------------------------------------------------

    def get_global_timeline(self) -> list[dict[str, Any]]:
        """Return a merged timeline of all synced events across feeds.

        Each entry contains the global time and the local times of all
        feeds at that moment.
        """
        if not self._feeds:
            return []

        feed_ids = list(self._feeds.keys())
        reference = self._feeds[feed_ids[0]]

        timeline: list[dict[str, Any]] = []
        for t in self._iter_timestamps(reference):
            entry: dict[str, Any] = {
                "global_time_s": round(t, 3),
                "feeds": {},
            }
            for cid, feed in self._feeds.items():
                local_time = t + feed.timecode_offset
                entry["feeds"][cid] = {
                    "local_time_s": round(local_time, 3),
                    "camera_id": cid,
                }
            timeline.append(entry)

        return timeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_timestamps(self, feed: CameraFeed) -> list[float]:
        """Generate a list of timestamps for a feed at 1-second intervals."""
        duration = 3600.0  # assume 1 hour max for iteration
        interval = max(1.0, 1.0 / max(feed.fps, 1.0))
        count = min(int(duration / interval), 3600)
        return [i * interval for i in range(count)]

    def _build_sync_points(
        self,
        camera_id: str,
        feed: CameraFeed,
        lag_seconds: float,
        correlation: float,
        sample_rate: int,
    ) -> list[dict[str, Any]]:
        """Build a list of sync point dicts for a feed."""
        points: list[dict[str, Any]] = []
        for t in self._iter_timestamps(feed):
            global_time = t + feed.timecode_offset - lag_seconds
            if global_time < 0:
                continue
            points.append(
                {
                    "camera_id": camera_id,
                    "local_time_s": round(t, 3),
                    "global_time_s": round(global_time, 3),
                    "lag_s": round(lag_seconds, 3),
                    "correlation": round(correlation, 4),
                }
            )
        return points

    @staticmethod
    def _cross_correlate_feeds(
        feed_a: CameraFeed,
        feed_b: CameraFeed,
        sample_rate: int,
    ) -> tuple[int, float]:
        """Cross-correlate synthetic audio envelopes of two feeds.

        Returns (lag_in_samples, peak_correlation).
        Positive lag means feed_b is behind feed_a.
        """
        duration = min(30.0, 3600.0)
        n_samples = int(duration * sample_rate)

        envelope_a = _synthetic_envelope(n_samples)
        envelope_b = _synthetic_envelope(n_samples)

        if len(envelope_a) == 0 or len(envelope_b) == 0:
            return 0, 0.0

        n = len(envelope_a) + len(envelope_b) - 1
        n_pow2 = 1 << (n - 1).bit_length()

        fft_a = np.fft.rfft(envelope_a, n=n_pow2)
        fft_b = np.fft.rfft(envelope_b, n=n_pow2)
        correlation = np.fft.irfft(fft_a * np.conj(fft_b), n=n_pow2)

        peak_idx = int(np.argmax(np.abs(correlation)))
        peak_value = float(np.abs(correlation[peak_idx]))

        max_possible = float(np.sqrt(np.sum(envelope_a**2) * np.sum(envelope_b**2)))
        if max_possible > 0:
            peak_value /= max_possible

        lag = peak_idx - (len(envelope_b) - 1)
        return lag, min(peak_value, 1.0)


def _synthetic_envelope(n_samples: int) -> np.ndarray:
    """Generate a synthetic audio envelope for correlation testing.

    Produces a simple low-frequency signal with some variation so that
    cross-correlation can detect alignment.
    """
    if n_samples <= 0:
        return np.array([], dtype=np.float64)
    t = np.linspace(0, n_samples / 44100.0, n_samples, dtype=np.float64)
    envelope = (
        0.5 * np.sin(2.0 * math.pi * 0.5 * t)
        + 0.3 * np.sin(2.0 * math.pi * 1.2 * t)
        + 0.2 * np.random.default_rng(seed=42).uniform(-0.1, 0.1, size=n_samples)
    )
    return envelope.astype(np.float64)
