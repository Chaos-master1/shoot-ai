"""Render tracking overlay video for visual debugging.

Usage:
    python scripts/render_tracking_overlay.py \\
        --video france_sweden_15min.mp4 \\
        --tracking tracking_output/frames_compact.pkl

Draws tracked player bboxes, IDs, team colors, and ball trajectory.
Outputs annotated video at specified path.

Overlay modes:
    --mode basic       Default mode: bboxes + IDs + ball trail
    --mode heatmap     Blends team heat maps over video frames
    --mode tactical    Shows pass arrows, formation labels, possession %
    --mode metrics     Overlays real-time metrics (player count, ball speed, MOTA/MOTP)

Features flag (comma-separated, default: "bbox,id,ball"):
    bbox, id, ball, heatmap, passes, metrics
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_tracking(tracking_dir: str | Path) -> dict[str, Any]:
    td = Path(tracking_dir)
    frames_path = td / "frames_compact.pkl"
    summary_path = td / "track_summary.json"
    ball_path = td / "ball_tracking.json"
    stats_path = td / "track_stats.json"
    mot_path = td / "mot_metrics.json"

    frames: list = []
    if frames_path.exists():
        with open(frames_path, "rb") as f:
            frames = pickle.load(f)
        print(f"Loaded {len(frames)} frame samples")

    summary: dict = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"Summary: {summary.get('n_tracks', 0)} tracks")

    ball_data: list = []
    if ball_path.exists():
        with open(ball_path) as f:
            ball_data = json.load(f)
        print(f"Ball data: {len(ball_data)} detections")

    track_stats: list = []
    if stats_path.exists():
        with open(stats_path) as f:
            track_stats = json.load(f)

    mot_metrics: dict = {}
    if mot_path.exists():
        with open(mot_path) as f:
            mot_metrics = json.load(f)

    team_by_tid: dict[int, str] = {}
    for ts in track_stats:
        tid = ts.get("track_id")
        team = ts.get("team", "?")
        if tid is not None:
            team_by_tid[tid] = team

    return {
        "frames": frames,
        "summary": summary,
        "ball_data": ball_data,
        "team_by_tid": team_by_tid,
        "track_stats": track_stats,
        "mot_metrics": mot_metrics,
    }


def get_team_color(team: str) -> tuple[int, int, int]:
    return {
        "home": (0, 200, 0),
        "away": (0, 0, 200),
        "referee": (200, 200, 0),
        "?": (128, 128, 128),
    }.get(team, (128, 128, 128))


def load_heatmaps(tracking_dir: Path, w: int, h: int) -> dict[str, np.ndarray]:
    heatmaps: dict[str, np.ndarray] = {}
    for team in ("home", "away"):
        path = tracking_dir / f"heatmap_{team}.png"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (w, h))
                heatmaps[team] = img
                print(f"Loaded heatmap: {path}")
            else:
                print(f"Warning: could not read heatmap {path}")
        else:
            print(f"Warning: heatmap not found: {path}")
    return heatmaps


def estimate_possession(
    frame_idx: int,
    ball_by_frame: dict[int, tuple[float, float]],
    frame_samples: list,
    team_by_tid: dict[int, str],
    possession_window: int = 300,
) -> dict[str, float]:
    """Estimate possession % by checking which team's players are nearest to ball."""
    home_frames = 0
    away_frames = 0
    start = max(0, frame_idx - possession_window)
    for f in range(start, frame_idx + 1):
        ball_pos = ball_by_frame.get(f)
        if ball_pos is None:
            continue
        # Find sample for this frame
        sample = None
        for s in frame_samples:
            if s["frame"] == f:
                sample = s
                break
        if not sample:
            continue
        nearest_team: str | None = None
        nearest_dist = float("inf")
        for det in sample.get("detections", []):
            bbox = det[0]
            cls_name = det[2]
            tid = det[3]
            if cls_name == "sports ball" or tid is None:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = np.hypot(cx - ball_pos[0], cy - ball_pos[1])
            team = team_by_tid.get(tid, "?")
            if dist < nearest_dist and team in ("home", "away"):
                nearest_dist = dist
                nearest_team = team
        if nearest_team == "home":
            home_frames += 1
        elif nearest_team == "away":
            away_frames += 1
    total = home_frames + away_frames
    if total == 0:
        return {"home": 50.0, "away": 50.0}
    return {"home": home_frames / total * 100, "away": away_frames / total * 100}


def render_heatmap_overlay(
    frame: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    alpha: float = 0.35,
):
    """Blend heatmap images over the video frame."""
    h, w = frame.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if "home" in heatmaps:
        hm = cv2.cvtColor(heatmaps["home"], cv2.COLOR_GRAY2BGR)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 1.0, hm, alpha, 0)  # type: ignore[assignment]
    if "away" in heatmaps:
        hm = cv2.cvtColor(heatmaps["away"], cv2.COLOR_GRAY2BGR)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(overlay, 1.0, hm, alpha, 0)  # type: ignore[assignment]
    return cv2.addWeighted(frame, 1.0, overlay, alpha, 0)


def render_tactical_annotations(
    frame: np.ndarray,
    frame_idx: int,
    ball_by_frame: dict[int, tuple[float, float]],
    team_by_tid: dict[int, str],
    frame_samples: list,
    pass_history: deque,
    formation_labels: tuple[str, str] = ("4-4-2", "4-3-3"),
):
    """Draw pass arrows, formation labels, and possession % overlay."""
    # Pass arrows: connect recent ball positions (skip if gap > 5 frames)
    pos_now = ball_by_frame.get(frame_idx)
    if pos_now:
        pass_history.append(pos_now)
    if len(pass_history) >= 2:
        # Determine arrow color: green for home, red for away
        possession = estimate_possession(frame_idx, ball_by_frame, frame_samples, team_by_tid)
        is_home = possession.get("home", 50) >= possession.get("away", 50)
        arrow_color = (0, 200, 0) if is_home else (0, 0, 200)
        prev = pass_history[-2]
        curr = pass_history[-1]
        pt1 = (int(prev[0]), int(prev[1]))
        pt2 = (int(curr[0]), int(curr[1]))
        cv2.arrowedLine(frame, pt1, pt2, arrow_color, 2, tipLength=0.3)

    # Formation labels (top-right corner)
    h = frame.shape[0]
    home_fmt, away_fmt = formation_labels
    cv2.putText(
        frame, f"Home: {home_fmt}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
    )
    cv2.putText(
        frame, f"Away: {away_fmt}", (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2
    )

    # Possession % overlay (top-center area)
    possession = estimate_possession(frame_idx, ball_by_frame, frame_samples, team_by_tid)
    pct_text = f"Poss: H {possession.get('home', 50):.0f}% - A {possession.get('away', 50):.0f}%"
    (tw, th), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cx = frame.shape[1] // 2
    cv2.putText(
        frame, pct_text, (cx - tw // 2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )


def render_metrics_overlay(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    ball_by_frame: dict[int, tuple[float, float]],
    current_sample: dict | None,
    mot_metrics: dict,
):
    """Overlay real-time metrics: player count, ball speed, FPS, MOT metrics."""
    h, w = frame.shape[:2]

    # FPS
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # Player count
    n_players = 0
    if current_sample:
        n_players = sum(1 for d in current_sample.get("detections", []) if d[2] != "sports ball")
    cv2.putText(
        frame,
        f"Players: {n_players}",
        (w - 160, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Ball speed (px/frame from consecutive positions)
    bx = ball_by_frame.get(frame_idx)
    bx_prev = ball_by_frame.get(frame_idx - 1)
    if bx and bx_prev:
        speed = np.hypot(bx[0] - bx_prev[0], bx[1] - bx_prev[1])
        cv2.putText(
            frame,
            f"Ball speed: {speed:.0f} px/f",
            (w - 160, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # MOT metrics if available
    if mot_metrics:
        y_off = 105
        for key in ("MOTA", "MOTP", "IDF1"):
            val = mot_metrics.get(key)
            if val is not None:
                cv2.putText(
                    frame,
                    f"{key}: {val:.2f}",
                    (w - 160, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_off += 25

    # Frame number
    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (w - 160, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )


def render(
    video_path: str,
    tracking_data: dict[str, Any],
    output_path: str,
    max_frames: int = 0,
    show_ball_trail: bool = True,
    features: list[str] | None = None,
    alpha: float = 0.35,
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 runtime attr absent from stale stubs
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if features is None:
        features = ["bbox", "id", "ball"]

    # Build frame->ball positions map
    ball_by_frame: dict[int, tuple[float, float]] = {}
    for b in tracking_data["ball_data"]:
        ball_by_frame[b["frame"]] = (b["x"], b["y"])

    team_colors: dict[int, tuple[int, int, int]] = {}
    for ts in tracking_data["track_stats"]:
        tid = ts.get("track_id")
        team = ts.get("team", "?")
        if tid is not None:
            team_colors[tid] = get_team_color(team)

    # Preload heatmaps if needed
    heatmaps: dict[str, np.ndarray] = {}
    if "heatmap" in features:
        heatmaps = (
            load_heatmaps(Path(tracking_data.get("_source", "")), w, h)
            if tracking_data.get("_source")
            else {}
        )

    # Pass history for tactical mode
    pass_history: deque = deque(maxlen=10)

    # Track possession for ball possession overlay
    frame_to_sample: dict[int, dict] = {}
    for s in tracking_data.get("frames", []):
        frame_to_sample[s["frame"]] = s

    frame_idx = 0
    sample_idx = 0
    rendered = 0

    print(f"Rendering {total_frames} frames to {output_path} [features: {','.join(features)}]...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Find sample data for this frame
        while (
            sample_idx < len(tracking_data["frames"])
            and tracking_data["frames"][sample_idx]["frame"] < frame_idx
        ):
            sample_idx += 1

        current_sample = None
        if (
            sample_idx < len(tracking_data["frames"])
            and tracking_data["frames"][sample_idx]["frame"] == frame_idx
        ):
            current_sample = tracking_data["frames"][sample_idx]

        # --- Feature: heatmap ---
        if "heatmap" in features and heatmaps:
            # Lazy-load heatmaps if not yet loaded
            if not heatmaps and tracking_data.get("_source"):
                heatmaps.update(load_heatmaps(Path(tracking_data["_source"]), w, h))
            if heatmaps:
                frame = render_heatmap_overlay(frame, heatmaps, alpha)

        # --- Feature: bbox and id ---
        if ("bbox" in features or "id" in features) and current_sample:
            for det in current_sample.get("detections", []):
                bbox = det[0]
                _ = det[1]
                cls_name = det[2]
                tid = det[3]
                if cls_name == "sports ball":
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = team_colors.get(tid, (128, 128, 128))
                if "bbox" in features:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if "id" in features and tid is not None:
                    label = f"#{tid}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Feature: ball ---
        if "ball" in features:
            if frame_idx in ball_by_frame:
                bx, by = ball_by_frame[frame_idx]
                bx, by = int(bx), int(by)
                cv2.circle(frame, (bx, by), 4, (0, 255, 255), -1)

            if show_ball_trail:
                trail = [
                    ball_by_frame.get(f)
                    for f in range(max(frame_idx - 30, 0), frame_idx + 1, 3)
                    if f in ball_by_frame
                ]
                for i in range(1, len(trail)):
                    p0 = trail[i - 1]
                    p1 = trail[i]
                    if p0 and p1:
                        pt1 = (int(p0[0]), int(p0[1]))
                        pt2 = (int(p1[0]), int(p1[1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

        # --- Feature: passes (tactical) ---
        if "passes" in features:
            render_tactical_annotations(
                frame,
                frame_idx,
                ball_by_frame,
                tracking_data["team_by_tid"],
                tracking_data.get("frames", []),
                pass_history,
            )

        # --- Feature: metrics ---
        if "metrics" in features:
            render_metrics_overlay(
                frame,
                frame_idx,
                fps,
                ball_by_frame,
                current_sample,
                tracking_data.get("mot_metrics", {}),
            )

        # Info overlay (always visible)
        cv2.putText(
            frame,
            f"Frame: {frame_idx}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        time_s = frame_idx / fps
        cv2.putText(
            frame,
            f"Time: {int(time_s // 60)}:{int(time_s % 60):02d}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        out.write(frame)
        rendered += 1
        frame_idx += 1

        if max_frames and rendered >= max_frames:
            break

        if rendered % 500 == 0:
            print(f"  Rendered {rendered}/{min(total_frames, max_frames or total_frames)}")

    cap.release()
    out.release()
    print(f"Done. {rendered} frames written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render tracking overlay video")
    parser.add_argument("--video", type=str, required=True, help="Input video")
    parser.add_argument(
        "--tracking", type=str, default="tracking_output", help="Tracking output directory"
    )
    parser.add_argument(
        "--output", type=str, default="tracking_overlay.mp4", help="Output video path"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames")
    parser.add_argument("--no-ball-trail", action="store_true", help="Hide ball trail")
    parser.add_argument(
        "--features",
        type=str,
        default="bbox,id,ball",
        help="Comma-separated overlay features: bbox,id,ball,heatmap,passes,metrics",
    )
    parser.add_argument(
        "--heatmap-alpha", type=float, default=0.35, help="Heatmap blend alpha (default: 0.35)"
    )
    args = parser.parse_args()

    tracking = load_tracking(args.tracking)

    # Store source dir for heatmap lazy loading
    tracking["_source"] = args.tracking

    features = [f.strip() for f in args.features.split(",") if f.strip()]

    render(
        args.video,
        tracking,
        args.output,
        max_frames=args.max_frames,
        show_ball_trail=not args.no_ball_trail,
        features=features,
        alpha=args.heatmap_alpha,
    )


if __name__ == "__main__":
    main()
