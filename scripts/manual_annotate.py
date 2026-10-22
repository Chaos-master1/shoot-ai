#!/usr/bin/env python3
"""Lightweight manual annotation tool for player positions.

Usage:
    python scripts/manual_annotate.py --video France_Sweden_clip_2min.mp4 --output data/ground_truth/manual_frames.json

Controls:
    Left-click: mark player position
    Right-click: undo last mark
    n: next frame
    b: previous frame
    q: quit and save
    c: clear current frame marks
"""

import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="France_Sweden_clip_2min.mp4")
    parser.add_argument("--output", default="data/ground_truth/manual_frames.json")
    parser.add_argument("--frame-skip", type=int, default=6, help="Detection frame skip")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=12,
        help="Sample every Nth detection frame (so every N*skip video frames)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=20, help="Max frames to show for annotation"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute sampled detection frames
    all_det_frames = list(range(0, total_frames, args.frame_skip))
    sampled = all_det_frames[:: args.sample_every][: args.max_frames]

    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    print(f"Detection frames: {len(all_det_frames)}")
    print(f"Sampling every {args.sample_every}th = {len(sampled)} frames to annotate")
    print("Controls: Left-click=mark, Right-click=undo, n=next, b=prev, c=clear, q=quit")

    annotations = {}  # {frame_num: [{'x': x, 'y': y}, ...]}
    current_idx = 0

    window_name = "Annotate Players (click to mark, n=next, q=quit)"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            fn = sampled[current_idx]
            if fn not in annotations:
                annotations[fn] = []
            annotations[fn].append({"x": float(x), "y": float(y)})

    cv2.setMouseCallback(window_name, mouse_callback)

    while current_idx < len(sampled):
        frame_num = sampled[current_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw existing marks
        if frame_num in annotations:
            for pt in annotations[frame_num]:
                cv2.circle(display, (int(pt["x"]), int(pt["y"])), 6, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    f"({int(pt['x'])},{int(pt['y'])})",
                    (int(pt["x"]) + 10, int(pt["y"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Info overlay
        info = f"Frame {frame_num}/{total_frames} ({current_idx + 1}/{len(sampled)}) - {len(annotations.get(frame_num, []))} marks"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            display,
            "n=next b=prev c=clear q=quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("n") or key == ord(" "):
            current_idx += 1
        elif key == ord("b"):
            current_idx = max(0, current_idx - 1)
        elif key == ord("c"):
            annotations[frame_num] = []
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "video": args.video,
                "frame_skip": args.frame_skip,
                "total_annotated": len(annotations),
                "total_points": sum(len(v) for v in annotations.values()),
                "annotations": {str(k): v for k, v in annotations.items()},
            },
            f,
            indent=2,
        )

    print(
        f"\nSaved {len(annotations)} frames with {sum(len(v) for v in annotations.values())} total points to {output_path}"
    )

    # Compare with tracking output if available
    tracking_path = Path("tracking_output_clip/track_summary.json")
    if tracking_path.exists():
        print("\nComparison with tracking output:")
        with open(tracking_path) as f:
            tracking = json.load(f)
        print(f"  Tracking tracks: {len(tracking.get('tracks', {}))}")
        print(f"  Manual frames: {len(annotations)}")
        print(f"  Manual points: {sum(len(v) for v in annotations.values())}")
        # Simple match rate per frame
        for fn_str, pts in sorted(annotations.items(), key=lambda x: int(x[0])):
            fn = int(fn_str)
            print(f"    Frame {fn}: {len(pts)} markers")


if __name__ == "__main__":
    main()
