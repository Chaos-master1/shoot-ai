"""Segment a broadcast match video to extract only the match portion."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np


def quick_analyze(video_path: Path, sample_sec: float = 2.0) -> list[dict]:
    """Quickly sample green pitch % at intervals using cv2 seeking."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {total} frames, {fps:.1f} FPS, {total / fps / 60:.1f} min")

    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])

    step = int(fps * sample_sec)
    samples = []
    t0 = time.time()

    for frame_idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pct = (np.count_nonzero(mask) / (frame.shape[0] * frame.shape[1])) * 100
        samples.append({"timestamp": frame_idx / fps, "green_pct": green_pct})
        if len(samples) % 100 == 0:
            pct = (frame_idx / total) * 100
            print(f"    {pct:.0f}% ({len(samples)} samples)", flush=True)

    cap.release()
    print(f"  Done in {time.time() - t0:.0f}s - {len(samples)} samples")
    return samples


def find_boundaries(samples: list[dict], green_thresh: float = 25.0) -> tuple[float, float]:
    """Find match start/end from green % curve."""
    import numpy as np

    greens = np.array([s["green_pct"] for s in samples])
    times = np.array([s["timestamp"] for s in samples])

    print(f"  Green %: min={greens.min():.1f}%, mean={greens.mean():.1f}%, max={greens.max():.1f}%")

    above = greens > green_thresh
    window = max(1, int(30 / (times[1] - times[0]) if len(times) > 1 else 1))
    smooth = np.convolve(above.astype(float), np.ones(window) / window, mode="valid")

    match_start = 0.0
    match_end = times[-1] if len(times) > 0 else 0

    for i in range(len(smooth)):
        if smooth[i] >= 0.5 and match_start == 0:
            match_start = times[i]
            print(f"  Match start: {match_start:.0f}s ({match_start / 60:.1f} min)")
            break

    for i in range(len(smooth) - 1, -1, -1):
        if smooth[i] >= 0.5:
            match_end = times[i + window] if i + window < len(times) else times[-1]
            print(f"  Match end: {match_end:.0f}s ({match_end / 60:.1f} min)")
            break

    # Detect halftime dip
    mid_start = len(smooth) // 3
    mid_end = 2 * len(smooth) // 3
    dips = []
    for i in range(mid_start, mid_end):
        if smooth[i] < 0.3:
            dips.append(times[i])
    if dips:
        halftime_start = dips[0]
        halftime_end = dips[-1]
        halflen = halftime_end - halftime_start
        if 300 < halflen < 1500:
            print(
                f"  Halftime: {halftime_start:.0f}s-{halftime_end:.0f}s ({halflen:.0f}s = {halflen / 60:.1f} min)"
            )
            fh = halftime_start - match_start
            sh = match_end - halftime_end
            print(
                f"  First half: {fh / 60:.1f} min, Second half: {sh / 60:.1f} min, Play: {(fh + sh) / 60:.1f} min"
            )

    return match_start, match_end


def extract(video_path: Path, start: float, end: float) -> Path:
    """Extract segment with ffmpeg."""
    out = video_path.parent / f"{video_path.stem}_match{video_path.suffix}"
    dur = end - start
    print(f"\n  Extracting {start:.0f}s -> {end:.0f}s ({dur / 60:.1f} min)")
    subprocess.run(
        [
            "ffmpeg",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(dur),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-y",
            str(out),
        ],
        check=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    print(f"  Saved: {out}")
    cap2 = cv2.VideoCapture(str(out))
    f = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cap2.release()
    print(f"  Output: {f} frames, {f / dur:.1f} FPS, {dur / 60:.1f} min")
    return out


def main():
    video = Path(__file__).resolve().parent.parent / "France vs Sweden.mp4"
    if not video.exists():
        print(f"Not found: {video}")
        return

    import argparse

    parser = argparse.ArgumentParser(description="Segment match video")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    args = parser.parse_args()

    if args.start is not None and args.end is not None:
        start, end = args.start, args.end
        print(
            f"  Using provided boundaries: {start:.0f}s -> {end:.0f}s ({(end - start) / 60:.1f} min)"
        )
    else:
        samples = quick_analyze(video, sample_sec=2.0)
        if not samples:
            print("No samples!")
            return
        start, end = find_boundaries(samples)
        print(f"  Boundaries: {start:.0f}s -> {end:.0f}s ({(end - start) / 60:.1f} min)")

    if end - start < 3000:
        print("  Skipping extraction (boundaries too narrow).")
        return

    extract(video, start, end)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
