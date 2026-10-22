"""Extract highlight clips from match video around key events.

Uses ffmpeg stream copy for fast extraction.
Events can come from tracking pipeline output, transcript parsing, or manual list.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    padding: float = 0.0,
) -> bool:
    """Extract a video segment using ffmpeg stream copy.

    Args:
        video_path: Source video.
        output_path: Output clip path.
        start_time: Start in seconds.
        end_time: End in seconds.
        padding: Extra seconds on each side.

    Returns:
        True on success.
    """
    start = max(0.0, start_time - padding)
    duration = end_time - start_time + padding * 2
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.2f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.2f}",
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return True
    except Exception as e:
        print(f"  ffmpeg failed: {e}")
        return False


def extract_highlights(
    video_path: Path,
    output_dir: Path,
    events: list[dict[str, Any]],
    padding: float = 15.0,
    min_gap: float = 10.0,
) -> list[Path]:
    """Extract clips around events.

    Args:
        video_path: Match video file.
        output_dir: Directory for clips.
        events: List of {type, timestamp, title, confidence}.
        padding: Seconds before/after each event.
        min_gap: If events are closer than this, merge them.

    Returns:
        List of output clip paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = _merge_nearby_events(events, min_gap)
    clips: list[Path] = []

    for i, evt in enumerate(merged):
        t = evt["timestamp"]
        start = max(0.0, t - padding)
        end = t + padding
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in evt["title"])
        safe_name = safe_name.strip().replace(" ", "_")[:48] or f"event_{i}"
        out_path = output_dir / f"{i + 1:02d}_{safe_name}.mp4"

        print(f"  [{i + 1}/{len(merged)}] {evt['title']} @ {t:.1f}s -> {out_path.name}")
        if extract_clip(video_path, out_path, start, end, padding=0):
            clips.append(out_path)

    print(f"Extracted {len(clips)}/{len(merged)} clips to {output_dir}")
    return clips


def _merge_nearby_events(
    events: list[dict[str, Any]],
    min_gap: float = 10.0,
) -> list[dict[str, Any]]:
    """Merge nearby events into single highlight moments."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    merged = [dict(sorted_events[0])]
    for evt in sorted_events[1:]:
        gap = evt["timestamp"] - merged[-1]["timestamp"]
        if gap < min_gap:
            merged[-1]["end_time"] = max(
                merged[-1].get("end_time", merged[-1]["timestamp"]), evt["timestamp"]
            )
            merged[-1]["title"] = f"{merged[-1]['title']} + {evt['title']}"
        else:
            merged.append(dict(evt))
    return merged


def detect_events_from_video(
    video_path: Path,
    goal_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect candidate events from video intensity analysis.

    Uses frame-diff peaks as a proxy for eventful moments
    (camera cuts, replays, celebrations) when no tracking data is available.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    events: list[dict[str, Any]] = []
    prev_gray = None
    frame_idx = 0
    sample_step = int(fps)  # 1 sample per second

    while frame_idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.norm(gray, prev_gray, cv2.NORM_L2) / gray.size
            if diff > 80:
                events.append(
                    {
                        "type": "intensity_peak",
                        "timestamp": frame_idx / fps,
                        "title": f"Activity @ {frame_idx // fps:.0f}s",
                        "confidence": min(1.0, diff / 200),
                    }
                )
        prev_gray = gray
        frame_idx += sample_step

    cap.release()
    return _merge_nearby_events(events, min_gap=10.0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract highlight clips from match video")
    parser.add_argument("video", type=Path, help="Match video file")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("highlights"), help="Output directory"
    )
    parser.add_argument("--padding", type=float, default=15.0, help="Seconds around each event")
    parser.add_argument(
        "--events", type=Path, default=None, help="JSON events file [{type, timestamp, title}]"
    )
    parser.add_argument(
        "--detect", action="store_true", help="Auto-detect events from video activity"
    )
    args = parser.parse_args()

    events = []
    if args.events:
        with open(args.events) as f:
            events = json.load(f)
    if args.detect:
        print("Auto-detecting events from video activity...")
        detected = detect_events_from_video(args.video)
        events.extend(detected)
        print(f"  Found {len(detected)} candidate moments")

    if not events:
        print("No events provided. Use --events <json> or --detect")
        return

    print(f"Extracting {len(events)} highlights from {args.video.name}...")
    extract_highlights(args.video, args.output_dir, events, padding=args.padding)


if __name__ == "__main__":
    main()
