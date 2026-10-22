"""Visual tracking validation: real footage in, annotated overlay video out.

Usage:
    PYTHONPATH=src .venv-linux/bin/python scripts/validate_tracking.py [video]

Writes data/tracking_validation.mp4 (boxes color-coded by assigned team,
track IDs, per-frame player count) and prints the tracking-quality numbers
the CV service itself computes, so detection/tracking/team-assignment can
be judged with eyes AND numbers. Never edits app state; read-only on the
video.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from kawkab.services.cv_service import CVService  # noqa: E402

VIDEO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/real_match.mp4")
OUT = Path("data/tracking_validation.mp4")
COLORS = {
    "home": (66, 217, 66),   # green (BGR)
    "away": (66, 66, 244),   # red
    "ball": (60, 200, 250),  # yellow
    "u": (190, 190, 190),    # gray = unassigned
}


def draw_text(img, text, org, color=(255, 255, 255), scale=0.45):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


async def main() -> int:
    svc = CVService(model_size="n", gpu_enabled=False)
    await svc.initialize()
    t0 = time.perf_counter()
    track = await svc.process_video(VIDEO, frame_skip=2)
    dt = time.perf_counter() - t0

    frames = track.frames
    per_frame_players = [
        sum(1 for d in f.detections if d.class_name == "person" and d.track_id is not None)
        for f in frames
    ]
    ball_frames = sum(1 for f in frames if any(d.class_name == "sports ball" for d in f.detections))
    teams = track.player_teams
    assigned = sum(1 for t in teams.values() if t in ("home", "away"))

    print(f"video: {VIDEO.name}  tracked in {dt:.1f}s (frame_skip=2)")
    print(f"frames with detections: {len(frames)}   unique tracks: {len(track.track_registry)}")
    if per_frame_players:
        print(
            "tracked players/frame: "
            f"mean={np.mean(per_frame_players):.1f} median={np.median(per_frame_players):.0f} "
            f"p90={np.percentile(per_frame_players, 90):.0f} (expect ~22)"
        )
    print(f"ball present in {ball_frames}/{len(frames)} frames ({100 * ball_frames / max(1, len(frames)):.0f}%)")
    print(f"team assignment: {assigned}/{len(teams)} tracks labeled home/away")
    m = track.tracking_metrics
    for key in ("raw_tracks_detected", "validated_player_tracks", "fragmentation_rate",
                "tracking_quality", "mot_self_consistency", "stitched_tracks"):
        if key in m:
            print(f"  {key}: {m[key]}")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx_map = {f.frame_number: f for f in frames}
    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fd = idx_map.get(fno)
        if fd is not None:
            for d in fd.detections:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                if d.class_name == "sports ball":
                    color = COLORS["ball"]
                elif d.track_id is not None:
                    color = COLORS.get(teams.get(d.track_id, "u"), COLORS["u"])
                else:
                    color = COLORS["u"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                if d.track_id is not None:
                    draw_text(frame, str(d.track_id), (x1, max(10, y1 - 3)), color)
            draw_text(frame, f"players:{per_frame_players[frames.index(fd)]}", (8, 16))
            draw_text(frame, f"frame:{fno}", (8, 30))
        writer.write(frame)
        fno += 1
    cap.release()
    writer.release()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) — watch it and judge the boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
