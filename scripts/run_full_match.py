"""Run France vs Sweden 15-min segment pipeline validation."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

FRANCE_SWEDEN_15 = Path(__file__).resolve().parent.parent / "france_sweden_15min.mp4"
FRANCE_SWEDEN_FULL = Path(__file__).resolve().parent.parent / "France vs Sweden.mp4"
FRANCE_SWEDEN_MATCH = Path(__file__).resolve().parent.parent / "France vs Sweden_match.mp4"


async def run(
    video_path: Path,
    frame_skip: int = 6,
    use_checkpoint: bool = False,
    resume: bool = False,
    tracker: str = "deepocsort",
):
    from kawkab.services.cv_service import CVService, PipelineCheckpoint

    print(f"=== Processing {video_path.name} ===")
    print("Resolution: checking...")
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    eff_fps = fps / (frame_skip + 1)
    print(f"Frames: {frames}, FPS: {fps}, Resolution: {w}x{h}")
    print(f"Frame skip: {frame_skip}, Effective FPS: {eff_fps:.1f}")
    est_min = (frames / max(fps, 1) / 60) * 2.0 * 7 / (frame_skip + 1) if fps > 0 else 0
    print(f"Estimated processing time: ~{est_min:.0f} min (GPU, RTX 4070)")

    checkpoint_interval = 500 if use_checkpoint else 0
    resume_state = None
    if resume:
        resume_state = PipelineCheckpoint.latest(video_path)
        if resume_state is None:
            print("No valid checkpoint found. Starting from scratch.")
        else:
            print(
                f"Resuming from checkpoint: frame={resume_state['frame_number']}, "
                f"det={resume_state['det_idx']}, "
                f"frames_in_checkpoint={len(resume_state['frames_compact'])}"
            )

    svc = CVService(model_size="m", gpu_enabled=True, tracker_type=tracker)
    await svc.initialize()

    match_data = await svc.process_video(
        video_path,
        frame_skip=frame_skip,
        enable_team_detection=True,
        checkpoint_interval=checkpoint_interval,
        resume_checkpoint=resume_state,
    )

    # Delete checkpoint on successful completion
    if match_data.checkpoint_manager is not None:
        match_data.checkpoint_manager.delete()
        print("Checkpoint deleted (pipeline completed successfully).")

    metrics = match_data.tracking_metrics

    # Save detailed output for downstream use and evaluation
    try:
        import json
        import pickle
        from datetime import datetime

        import numpy as np

        def _to_python(val):
            if isinstance(val, (np.float32, np.float64)):
                return float(val)
            if isinstance(val, (np.int32, np.int64)):
                return int(val)
            return val

        output_dir = video_path.parent / "tracking_output"
        output_dir.mkdir(exist_ok=True)

        # 1. Track summary
        events = []
        for fd in match_data.frames:
            for det in fd.detections:
                if det.class_name == "sports ball" and det.confidence > 0.5:
                    events.append(
                        {
                            "type": "ball",
                            "timestamp": fd.timestamp,
                            "frame": fd.frame_number,
                            "title": f"Ball @ {fd.timestamp:.0f}s",
                        }
                    )

        def team_label(tid):
            return match_data.player_teams.get(tid, "unclassified")

        summary = {
            "n_tracks": len(match_data.track_registry),
            "tracks": {
                str(tid): {
                    "frames": int(i["frames_tracked"]),
                    "pct": round(float(i["lifetime_pct"]), 2),
                    "team": team_label(tid),
                }
                for tid, i in match_data.track_registry.items()
            },
            "events_sample": events[:50],
            "metrics": {
                k: _to_python(v) for k, v in metrics.items() if not isinstance(v, (dict, list))
            },
            "team_detection": metrics.get("team_detection", {}),
            "run_timestamp": datetime.now().isoformat(),
            "video": str(video_path.name),
        }
        with open(output_dir / "track_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 2. Full frame data (sampled every 30 frames for evaluation)
        compact = []
        for idx, fd in enumerate(match_data.frames):
            if idx % 30 == 0:
                dets = []
                for d in fd.detections:
                    dets.append(
                        [
                            [_to_python(v) for v in d.bbox],
                            _to_python(d.confidence),
                            d.class_name,
                            d.track_id,
                        ]
                    )
                compact.append(
                    {"frame": fd.frame_number, "timestamp": fd.timestamp, "detections": dets}
                )
        with open(output_dir / "frames_compact.pkl", "wb") as f:
            pickle.dump(compact, f)

        # 3. Ball-only tracking (top 1000 ball detections)
        ball_data = []
        for fd in match_data.frames:
            for det in fd.detections:
                if det.class_name == "sports ball" and det.confidence > 0.3:
                    cx = _to_python((det.bbox[0] + det.bbox[2]) / 2)
                    cy = _to_python((det.bbox[1] + det.bbox[3]) / 2)
                    ball_data.append(
                        {
                            "frame": fd.frame_number,
                            "timestamp": fd.timestamp,
                            "x": cx,
                            "y": cy,
                            "conf": _to_python(det.confidence),
                        }
                    )
                    break
        with open(output_dir / "ball_tracking.json", "w") as f:
            json.dump(ball_data[:1000], f, indent=2, default=str)

        # 4. Per-track stats
        track_stats = []
        for tid, tdata in match_data.track_registry.items():
            track_stats.append(
                {
                    "track_id": tid,
                    "team": team_label(tid),
                    "frames_tracked": int(tdata.get("frames_tracked", 0)),
                    "lifetime_pct": round(float(tdata.get("lifetime_pct", 0)), 4),
                    "avg_confidence": round(float(tdata.get("confidence_avg", 0)), 3),
                }
            )
        with open(output_dir / "track_stats.json", "w") as f:
            json.dump(track_stats, f, indent=2, default=str)

        print(f"Output saved to {output_dir}")
        print(f"  - track_summary.json ({len(summary['tracks'])} tracks)")
        print(f"  - frames_compact.pkl ({len(compact)} frames)")
        print(f"  - ball_tracking.json ({len(ball_data)} ball detections)")
        print(f"  - track_stats.json ({len(track_stats)} entries)")
    except Exception as e:
        import traceback

        print(f"Output save error: {e}")
        traceback.print_exc()

    print()
    print("=== RESULTS ===")
    print(f"Raw tracks detected: {metrics.get('raw_tracks_detected')}")
    print(f"Validated player tracks: {metrics.get('validated_player_tracks')}")
    print(f"Fragmentation rate: {metrics.get('fragmentation_rate')}x")
    print(f"Tracking quality: {metrics.get('tracking_quality')}")
    print(f"Stitched tracks: {metrics.get('stitched_tracks')}")
    print(f"Stitch merges: {len(metrics.get('stitch_merge_map', {}))}")

    td = metrics.get("team_detection", {})
    if td:
        print(
            f"Team detection: home={td.get('home_size')}, away={td.get('away_size')}, ref={td.get('ref_size')}"
        )

    auto_h = metrics.get("auto_homography")
    print(f"Auto-calibration: {'yes' if auto_h else 'no'}")

    print()
    print(f"Match type: {match_data.match_type}")
    print(
        f"Duration: {match_data.duration_seconds:.0f}s ({match_data.duration_seconds / 60:.1f} min)"
    )
    print(f"Sampled frames: {len(match_data.frames)}")

    await svc.shutdown()
    print("=== Done ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pipeline on France vs Sweden match")
    parser.add_argument("--skip", type=int, default=6, help="Frame skip rate (default: 6)")
    parser.add_argument(
        "--full", action="store_true", help="Process full match instead of 15-min segment"
    )
    parser.add_argument(
        "--checkpoint", action="store_true", help="Enable periodic checkpoint saves"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Checkpoint save interval in detection frames (default: 500)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="deepocsort",
        choices=["deepocsort", "botsort", "bytetrack", "strongsort"],
        help="boxmot tracker backend (default: deepocsort)",
    )
    args = parser.parse_args()

    if args.full and FRANCE_SWEDEN_MATCH.exists():
        video = FRANCE_SWEDEN_MATCH
    elif args.full:
        video = FRANCE_SWEDEN_FULL
    else:
        video = FRANCE_SWEDEN_15
    asyncio.run(
        run(
            video,
            frame_skip=args.skip,
            use_checkpoint=args.checkpoint or args.resume,
            resume=args.resume,
            tracker=args.tracker,
        )
    )
