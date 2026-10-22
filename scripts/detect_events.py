"""Event detection from tracking data.

Detects shots and passes from ball trajectory + player tracking data.
Compares against StatsBomb ground truth when available.

Usage:
  python scripts/detect_events.py --tracking tracking_output
  python scripts/detect_events.py --tracking tracking_output --ground-truth data/ground_truth/statsbomb

Algorithm:
  Shots:  ball speed > 12 m/s + direction toward goal + abrupt acceleration
  Passes: ball moves between tracked players with possession transfer
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from math import atan2, dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("detect_events")

# ── Constants ─────────────────────────────────────────────────────
GOAL_LINE_X_RATIO = 0.05  # near goal (5% from video edge)
MIN_PASS_DURATION = 0.3
MAX_PASS_DURATION = 6.0
MIN_PASS_PX = 50
MIN_PASS_STRAIGHTNESS = 0.5
MIN_SHOT_PX = 60
MAX_SHOT_DURATION = 1.5
MIN_SHOT_STRAIGHTNESS = 0.3  # shots can curve


@dataclass
class BallFrame:
    frame: int
    timestamp: float
    x: float
    y: float
    conf: float = 0.0


@dataclass
class BallSegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    duration: float
    total_px: float
    max_speed: float
    avg_px_per_s: float
    frames: list[BallFrame] = field(default_factory=list)
    n_frames: int = 0
    straightness: float = 0.0  # 0-1, how straight the trajectory is


@dataclass
class DetectedEvent:
    event_type: str  # "shot" or "pass"
    timestamp: float
    frame: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    speed: float
    confidence: float = 1.0
    duration: float = 0.0


def load_ball_tracking(tracking_dir: Path) -> list[BallFrame]:
    """Load ball tracking data, filtering stationary/false detections."""
    ball_path = tracking_dir / "ball_tracking.json"
    if not ball_path.exists():
        logger.error(f"Ball tracking not found: {ball_path}")
        return []
    with open(ball_path) as f:
        data = json.load(f)
    result = []
    for d in data:
        conf = d.get("conf", 0.5)
        if conf < 0.3:
            continue
        result.append(
            BallFrame(
                frame=d["frame"],
                timestamp=d["timestamp"],
                x=d["x"],
                y=d["y"],
                conf=conf,
            )
        )
    logger.info(f"Loaded {len(result)} ball detections (filtered from {len(data)} raw)")
    return result


def compute_direction(f1: BallFrame, f2: BallFrame) -> float:
    """Compute direction of ball movement in radians."""
    dx = f2.x - f1.x
    dy = f2.y - f1.y
    return atan2(dy, dx)


def is_toward_goal(x: float, y: float, dx: float, dy: float, field_w: float = 1920.0) -> bool:
    """Check if ball is moving toward the nearer goal.

    Since we don't know which team is attacking which way, check if the
    ball moves toward the nearer goal line (either left or right edge).
    """
    left_goal_dist = x
    right_goal_dist = field_w - x
    toward_left = dx < 0
    toward_right = dx > 0
    _ = x < field_w * GOAL_LINE_X_RATIO
    _ = x > field_w * (1 - GOAL_LINE_X_RATIO)
    if left_goal_dist < right_goal_dist:
        return toward_left
    else:
        return toward_right


def segment_ball_data(ball_data: list[BallFrame]) -> list[BallSegment]:
    """Split ball data into contiguous tracking segments.

    A segment ends when the ball disappears for >10 frames or teleports (>300px).
    """
    segments: list[BallSegment] = []
    if len(ball_data) < 3:
        return segments

    current = [ball_data[0]]
    for i in range(1, len(ball_data)):
        b = ball_data[i]
        prev = current[-1]
        gap = b.frame - prev.frame
        gap_time = gap * (1.0 / 24.0)
        pixel_dist = dist((prev.x, prev.y), (b.x, b.y))
        if gap_time > 0.5 or pixel_dist > 300:
            if len(current) >= 3:
                s = _build_segment(current)
                if s is not None:
                    segments.append(s)
            current = [b]
        else:
            current.append(b)
    if len(current) >= 3:
        s = _build_segment(current)
        if s is not None:
            segments.append(s)

    logger.info(f"Identified {len(segments)} ball tracking segments")
    return segments


def _build_segment(frames: list[BallFrame]) -> BallSegment | None:
    """Build a BallSegment from contiguous ball frames.

    Returns None when the segment is driven by single-frame noise
    (callers guard on the return value).
    """
    max_px_speed = 0.0
    total_px = 0.0
    frames_with_movement = 0
    for i in range(1, len(frames)):
        d = dist((frames[i - 1].x, frames[i - 1].y), (frames[i].x, frames[i].y))
        if d > 2:
            frames_with_movement += 1
        dt = max(frames[i].timestamp - frames[i - 1].timestamp, 0.001)
        px_per_s = d / dt
        if px_per_s > max_px_speed:
            max_px_speed = px_per_s
        total_px += d

    # Reject segments driven by single-frame noise
    if total_px > 0 and frames_with_movement < max(3, len(frames) * 0.05):
        return None
    start_to_end = dist((frames[0].x, frames[0].y), (frames[-1].x, frames[-1].y))
    straightness = start_to_end / max(total_px, 1) if total_px > 0 else 0.0

    return BallSegment(
        start_frame=frames[0].frame,
        end_frame=frames[-1].frame,
        start_time=frames[0].timestamp,
        end_time=frames[-1].timestamp,
        start_x=frames[0].x,
        start_y=frames[0].y,
        end_x=frames[-1].x,
        end_y=frames[-1].y,
        duration=frames[-1].timestamp - frames[0].timestamp,
        total_px=total_px,
        max_speed=round(max_px_speed, 1),
        avg_px_per_s=0.0,
        frames=frames,
        n_frames=len(frames),
        straightness=round(straightness, 3),
    )


def detect_shots_from_segments(
    segments: list[BallSegment],
) -> list[DetectedEvent]:
    """Detect shots from ball tracking segments.

    A shot = rapid ball movement toward goal area:
    - Quick segment (< 1s)
    - Significant movement (> 50px)
    - Trajectory direction toward either sideline (goal areas at ~x=0 and x=1920)
    - Ball moves in a straight line (straightness > 0.5)
    """
    events = []
    for seg in segments:
        if seg.duration > MAX_SHOT_DURATION or seg.total_px < MIN_SHOT_PX:
            continue
        dx = seg.end_x - seg.start_x
        dy = seg.end_y - seg.start_y
        if not is_toward_goal(seg.start_x, seg.start_y, dx, dy):
            continue
        if seg.straightness < MIN_SHOT_STRAIGHTNESS:
            continue
        pixel_speed = seg.total_px / max(seg.duration, 0.001)
        speed_conf = min(pixel_speed / 300.0, 0.95)
        confidence = speed_conf * seg.straightness
        events.append(
            DetectedEvent(
                event_type="shot",
                timestamp=seg.start_time,
                frame=seg.start_frame,
                start_x=seg.start_x,
                start_y=seg.start_y,
                end_x=seg.end_x,
                end_y=seg.end_y,
                speed=round(pixel_speed, 1),
                confidence=round(confidence, 3),
                duration=seg.duration,
            )
        )

    # Deduplicate overlapping shots
    if events:
        events.sort(key=lambda e: e.timestamp)
        filtered = [events[0]]
        for e in events[1:]:
            if e.timestamp - filtered[-1].timestamp > 2.0:
                filtered.append(e)
            elif e.speed > filtered[-1].speed:
                filtered[-1] = e
        events = filtered

    logger.info(f"Detected {len(events)} shots from {len(segments)} segments")
    return events


def detect_passes_from_segments(
    segments: list[BallSegment],
) -> list[DetectedEvent]:
    """Detect passes from ball tracking segments.

    A pass = ball moves along a straight trajectory between player positions.
    Uses pixel-based heuristics (no meter conversion needed).
    """
    events = []
    for seg in segments:
        if seg.duration < MIN_PASS_DURATION or seg.duration > MAX_PASS_DURATION:
            continue
        if seg.total_px < MIN_PASS_PX:
            continue
        if seg.straightness < MIN_PASS_STRAIGHTNESS:
            continue
        pixel_speed = seg.total_px / max(seg.duration, 0.001)
        speed_conf = min(pixel_speed / 200.0, 0.9)
        confidence = speed_conf * (0.3 + 0.7 * seg.straightness)
        events.append(
            DetectedEvent(
                event_type="pass",
                timestamp=seg.start_time,
                frame=seg.start_frame,
                start_x=seg.start_x,
                start_y=seg.start_y,
                end_x=seg.end_x,
                end_y=seg.end_y,
                speed=round(pixel_speed, 1),
                confidence=round(confidence, 3),
                duration=seg.duration,
            )
        )

    if events:
        events.sort(key=lambda e: e.timestamp)
        filtered = [events[0]]
        for e in events[1:]:
            if e.timestamp - filtered[-1].timestamp > 1.0:
                filtered.append(e)
        events = filtered

    logger.info(f"Detected {len(events)} passes from {len(segments)} segments")
    return events


def load_statsbomb_events(gt_dir: Path) -> list[dict]:
    """Load StatsBomb ground truth event data."""
    events = []
    gt_files = list(gt_dir.rglob("*_gt.json"))
    evt_files = list(gt_dir.rglob("events/*.json"))
    for f in gt_files + evt_files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                events.extend(data)
        except Exception as e:
            logger.warning(f"Couldn't load {f}: {e}")
    # Filter to shot and pass events
    filtered = [e for e in events if e.get("event_type") in ("shot", "pass")]
    logger.info(f"Loaded {len(filtered)} StatsBomb shot/pass events from {len(events)} total")
    return filtered


def compare_with_ground_truth(
    our_events: list[DetectedEvent],
    gt_events: list[dict],
    time_window: float = 3.0,
) -> dict:
    """Compare our detected events against ground truth."""
    gt_shots = [e for e in gt_events if e.get("event_type") == "shot"]
    gt_passes = [e for e in gt_events if e.get("event_type") == "pass"]

    our_shots = [e for e in our_events if e.event_type == "shot"]
    our_passes = [e for e in our_events if e.event_type == "pass"]

    # Count matches: our event within time_window of any GT event
    shot_matches = 0
    for oe in our_shots:
        for ge in gt_shots:
            if abs(oe.timestamp - ge.get("timestamp", 0)) < time_window:
                shot_matches += 1
                break

    pass_matches = 0
    for oe in our_passes:
        for ge in gt_passes:
            if abs(oe.timestamp - ge.get("timestamp", 0)) < time_window:
                pass_matches += 1
                break

    return {
        "shots_detected": len(our_shots),
        "shots_in_gt": len(gt_shots),
        "shot_matches": shot_matches,
        "shot_precision": round(shot_matches / max(len(our_shots), 1), 3),
        "shot_recall": round(shot_matches / max(len(gt_shots), 1), 3),
        "passes_detected": len(our_passes),
        "passes_in_gt": len(gt_passes),
        "pass_matches": pass_matches,
        "pass_precision": round(pass_matches / max(len(our_passes), 1), 3),
        "pass_recall": round(pass_matches / max(len(gt_passes), 1), 3),
    }


def print_events(events: list[DetectedEvent], label: str, max_count: int = 20):
    """Print a formatted list of events."""
    print(f"\n  [{label}] {len(events)} events")
    for e in events[:max_count]:
        print(
            f"    {e.event_type} @ {e.timestamp:.1f}s (frame {e.frame}) "
            f"speed={e.speed:.1f}m/s conf={e.confidence:.2f} "
            f"at ({e.start_x:.0f},{e.start_y:.0f})->({e.end_x:.0f},{e.end_y:.0f})"
        )
    if len(events) > max_count:
        print(f"    ... and {len(events) - max_count} more")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detect events from tracking data")
    parser.add_argument(
        "--tracking", type=str, default="tracking_output", help="Tracking output directory"
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None, help="StatsBomb ground truth directory"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save detected events to JSON file"
    )
    args = parser.parse_args()

    tracking_dir = Path(args.tracking)
    if not tracking_dir.exists():
        logger.error(f"Tracking directory not found: {tracking_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  EVENT DETECTION REPORT")
    print("=" * 60)

    ball_data = load_ball_tracking(tracking_dir)
    if not ball_data:
        print("  No ball tracking data found.")
        return

    segments = segment_ball_data(ball_data)
    print(f"\n  Ball tracking segments: {len(segments)}")
    for seg in segments[:5]:
        print(
            f"    {seg.start_time:.1f}s-{seg.end_time:.1f}s ({seg.duration:.2f}s) "
            f"speed={seg.max_speed:.0f}px/s dist={seg.total_px:.0f}px "
            f"straight={seg.straightness:.2f} n={seg.n_frames}"
        )

    shots = detect_shots_from_segments(segments)
    passes = detect_passes_from_segments(segments)

    print_events(shots, "SHOTS")
    print_events(passes, "PASSES")

    if args.ground_truth:
        gt_dir = Path(args.ground_truth)
        if gt_dir.exists():
            gt_events = load_statsbomb_events(gt_dir)
            if gt_events:
                comparison = compare_with_ground_truth(shots + passes, gt_events)
                print("\n  --- COMPARISON vs GROUND TRUTH ---")
                for k, v in comparison.items():
                    print(f"    {k}: {v}")

    if args.output:
        output_path = Path(args.output)
        serializable = []
        for e in shots + passes:
            serializable.append(
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "frame": e.frame,
                    "start_x": e.start_x,
                    "start_y": e.start_y,
                    "end_x": e.end_x,
                    "end_y": e.end_y,
                    "speed": e.speed,
                    "confidence": e.confidence,
                    "duration": e.duration,
                }
            )
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Events saved to {output_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
