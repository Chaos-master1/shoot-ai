"""CLI entry point for Kawkab AI — desktop app launcher + tracking pipeline.

Usage:
    python -m kawkab                     # launch the desktop app (default, same as `gui`)
    python -m kawkab gui                 # launch the desktop app explicitly
    python -m kawkab track --video match.mp4 [--output tracking_output] [--skip 6]
    python -m kawkab track --pattern "*.mp4" --input-dir videos/ [--output-dir tracking_output]
    python -m kawkab batch --pattern "*.mp4" --input-dir videos/ [--output-dir batches] [--skip 6]
    python -m kawkab evaluate --tracking tracking_output --video match.mp4
    python -m kawkab render --video match.mp4 --tracking tracking_output [--output overlay.mp4]
    python -m kawkab events --tracking tracking_output [--output events.json]
    python -m kawkab e2e --match-id 1
    python -m kawkab benchmark [--module xg_model] [--iterations 5] [--output benchmark_results.json]
    python -m kawkab validate [--corpus data/statsbomb_corpus] [--out docs/validation]
    python -m kawkab train-yolo --data dataset.yaml [--epochs 100]
    python -m kawkab prepare-data --source raw_annotations --output data/soccer_net
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Professional football tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # gui (default when no subcommand is given)
    subparsers.add_parser("gui", help="Launch the Kawkab AI desktop application (default)")

    # track
    track_p = subparsers.add_parser("track", help="Run tracking on video(s)")
    track_p.add_argument("--video", type=str, default=None, help="Input video path (single)")
    track_p.add_argument(
        "--output", type=str, default="tracking_output", help="Output directory for single video"
    )
    track_p.add_argument(
        "--pattern", type=str, default=None, help="Glob pattern (e.g. '*.mp4') for multi-video"
    )
    track_p.add_argument(
        "--input-dir", type=str, default=".", help="Input directory for --pattern glob"
    )
    track_p.add_argument(
        "--output-dir", type=str, default=None, help="Base output directory for multi-video"
    )
    track_p.add_argument("--skip", type=int, default=6, help="Frame skip rate")
    track_p.add_argument(
        "--tracker",
        type=str,
        default="deepocsort",
        choices=["deepocsort", "botsort", "bytetrack", "strongsort"],
    )
    track_p.add_argument("--checkpoint", action="store_true", help="Enable checkpoint saves")
    track_p.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # batch
    batch_p = subparsers.add_parser("batch", help="Batch process multiple videos")
    batch_p.add_argument("--pattern", type=str, default="*.mp4", help="Glob pattern (e.g. '*.mp4')")
    batch_p.add_argument("--input-dir", type=str, default=".", help="Input directory")
    batch_p.add_argument(
        "--output-dir", type=str, default="batch_output", help="Base output directory"
    )
    batch_p.add_argument("--skip", type=int, default=6, help="Frame skip rate")
    batch_p.add_argument(
        "--tracker",
        type=str,
        default="deepocsort",
        choices=["deepocsort", "botsort", "bytetrack", "strongsort"],
    )

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate tracking quality")
    eval_p.add_argument("--tracking", type=str, default="tracking_output", help="Tracking output")
    eval_p.add_argument("--video", type=str, help="Video path (for cut detection eval)")
    eval_p.add_argument("--ground-truth", type=str, help="Metrica ground truth CSV")

    # render
    render_p = subparsers.add_parser("render", help="Render tracking overlay video")
    render_p.add_argument("--video", type=str, required=True, help="Input video")
    render_p.add_argument("--tracking", type=str, default="tracking_output")
    render_p.add_argument("--output", type=str, default="tracking_overlay.mp4")
    render_p.add_argument("--max-frames", type=int, default=0)
    render_p.add_argument("--no-ball-trail", action="store_true", help="Hide ball trail")
    render_p.add_argument(
        "--features",
        type=str,
        default="bbox,id,ball",
        help="Comma-separated: bbox,id,ball,heatmap,passes,metrics",
    )
    render_p.add_argument(
        "--heatmap-alpha", type=float, default=0.35, help="Heatmap blend alpha (default: 0.35)"
    )

    # events
    events_p = subparsers.add_parser("events", help="Detect events from tracking")
    events_p.add_argument("--tracking", type=str, default="tracking_output")
    events_p.add_argument("--output", type=str, default="events.json")
    events_p.add_argument("--ground-truth", type=str, help="StatsBomb GT directory")

    # possession
    poss_p = subparsers.add_parser("possession", help="Extract possession chains from tracking")
    poss_p.add_argument(
        "--tracking", type=str, default="tracking_output", help="Tracking output directory"
    )
    poss_p.add_argument(
        "--events", type=str, default=None, help="Events JSON (optional, auto-detected if omitted)"
    )
    poss_p.add_argument("--home", type=str, default="home", help="Home team name")
    poss_p.add_argument("--away", type=str, default="away", help="Away team name")

    # link-players
    link_p = subparsers.add_parser("link-players", help="Cross-match player linking pipeline")
    link_p.add_argument(
        "--match", type=int, default=None, help="Single match ID (omit to process all matches)"
    )

    # e2e
    e2e_p = subparsers.add_parser("e2e", help="Run the full analytical E2E pipeline on a match")
    e2e_p.add_argument("--match-id", type=int, default=1, help="Match ID to analyze")
    e2e_p.add_argument("--output", type=str, default=None, help="Output directory for results")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run analytical module benchmarks")
    bench_p.add_argument(
        "--module", type=str, default=None, help="Specific module to benchmark (default: all)"
    )
    bench_p.add_argument("--iterations", type=int, default=3, help="Number of benchmark iterations")
    bench_p.add_argument(
        "--output", type=str, default=None, help="Save benchmark results to JSON file"
    )

    # validate (Phase 2 elite-readiness: reproducible model-validation report)
    val_p = subparsers.add_parser(
        "validate",
        help="Build the model-validation report (model cards + calibration)",
    )
    val_p.add_argument(
        "--corpus",
        type=str,
        default="data/statsbomb_corpus",
        help="StatsBomb open-data corpus directory",
    )
    val_p.add_argument(
        "--out",
        type=str,
        default="docs/validation",
        help="Output directory for validation_report.json/.md",
    )
    val_p.add_argument(
        "--max-matches", type=int, default=40, help="Cap on corpus matches per calibration section"
    )

    # import (season-scale vendor ingest: one command, whole season)
    imp_p = subparsers.add_parser(
        "import",
        help="Import a vendor match file or a whole directory as a season",
    )
    imp_p.add_argument(
        "path", type=str, help="StatsBomb event file, or directory of them (season import)"
    )
    imp_p.add_argument(
        "--competition",
        type=str,
        default=None,
        help="Competition tag written to every imported match",
    )
    imp_p.add_argument(
        "--season-id",
        type=int,
        default=None,
        help="Season id (seasons table) to attach to imported matches",
    )
    imp_p.add_argument(
        "--match-date",
        type=str,
        default=None,
        help="Fallback match date (YYYY-MM-DD) when the file carries none",
    )
    imp_p.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Cap on matches imported this run (smoke tests)",
    )

    # train-yolo
    train_p = subparsers.add_parser("train-yolo", help="Fine-tune YOLO on football data")
    train_p.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    train_p.add_argument("--base-model", type=str, default="yolo11m.pt", help="Base model")
    train_p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_p.add_argument("--batch", type=int, default=16, help="Batch size")
    train_p.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_p.add_argument("--device", type=str, default="0", help="CUDA device")
    train_p.add_argument(
        "--prepare", action="store_true", help="Run SoccerNet data preparation first"
    )

    # prepare-data
    prep_p = subparsers.add_parser("prepare-data", help="Prepare SoccerNet annotations for YOLO")
    prep_p.add_argument(
        "--source",
        type=str,
        default="data/ground_truth/skillcorner/opendata-master/data",
        help="SoccerNet tracking annotations directory",
    )
    prep_p.add_argument(
        "--output",
        type=str,
        default="data/soccer_net",
        help="Output directory for YOLO-format dataset",
    )

    args = parser.parse_args()
    if args.command is None or args.command == "gui":
        sys.exit(_run_gui())

    if args.command == "track":
        asyncio.run(_run_tracking(args))
    elif args.command == "batch":
        asyncio.run(_run_batch(args))
    elif args.command == "evaluate":
        _run_evaluation(args)
    elif args.command == "render":
        _run_render(args)
    elif args.command == "events":
        _run_events(args)
    elif args.command == "possession":
        _run_possession(args)
    elif args.command == "link-players":
        _run_link_players(args)
    elif args.command == "e2e":
        _run_e2e(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "validate":
        _run_validate(args)
    elif args.command == "import":
        _run_import(args)
    elif args.command == "train-yolo":
        _run_train_yolo(args)
    elif args.command == "prepare-data":
        _run_prepare_data(args)


def _run_gui() -> int:
    """Launch the Kawkab AI desktop application.

    This is the default action when `python -m kawkab` is run with no
    subcommand (and also available explicitly as `python -m kawkab gui`).
    The import is lazy so the lighter CLI subcommands (track/evaluate/
    benchmark/...) don't pay the cost of importing PySide6 + QtWebEngine.
    """
    from kawkab.app import run_app

    return run_app()


async def _process_single_video(video_path: str, output_dir: str, skip: int, tracker: str) -> dict:
    """Run tracking on one video and write summary. Returns result dict."""
    import json

    from kawkab.services.cv_service import CVService

    svc = CVService(model_size="m", gpu_enabled=True, tracker_type=tracker)
    await svc.initialize()
    try:
        match_data = await svc.process_video(Path(video_path), frame_skip=skip)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tracks = []
        for tid, tdata in match_data.track_registry.items():
            tracks.append(
                {
                    "track_id": tid,
                    "first_seen": tdata.get("first_seen"),
                    "last_seen": tdata.get("last_seen"),
                    "frames_tracked": tdata.get("frames_tracked"),
                    "confidence_avg": tdata.get("confidence_avg"),
                    "team": match_data.player_teams.get(tid, "?"),
                }
            )
        summary = {
            "video": Path(video_path).name,
            "n_tracks": len(tracks),
            "raw_tracks": match_data.tracking_metrics.get("raw_tracks_detected"),
            "fragmentation": match_data.tracking_metrics.get("fragmentation_rate"),
            "quality": match_data.tracking_metrics.get("tracking_quality"),
            "duration_s": match_data.duration_seconds,
            "team_detection": match_data.tracking_metrics.get("team_detection"),
            "tracks": tracks,
        }
        with open(out / "track_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(
            f"  OK: {Path(video_path).name} -> {out / 'track_summary.json'} ({len(tracks)} tracks)"
        )
        return {"video": Path(video_path).name, "status": "ok", "tracks": len(tracks)}
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"  FAIL: {Path(video_path).name}: {e}")
        return {"video": Path(video_path).name, "status": "failed", "error": str(e)}
    finally:
        await svc.shutdown()


async def _run_tracking(args):

    if args.video:
        # Single-video mode
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        result = await _process_single_video(args.video, args.output, args.skip, args.tracker)
        if result["status"] == "failed":
            sys.exit(1)
    elif args.pattern:
        # Multi-video mode
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        input_dir = Path(args.input_dir)
        videos = sorted(input_dir.glob(args.pattern))
        if not videos:
            print(f"No videos matching '{args.pattern}' in {input_dir}")
            sys.exit(1)
        base_out = args.output_dir or args.output
        print(f"Tracking {len(videos)} videos from {input_dir} (pattern: {args.pattern})")
        results = []
        for i, video in enumerate(videos):
            video_out_dir = Path(base_out) / video.stem
            print(f"[{i + 1}/{len(videos)}] Processing {video.name}...")
            r = await _process_single_video(str(video), str(video_out_dir), args.skip, args.tracker)
            results.append(r)
        ok = [r for r in results if r["status"] == "ok"]
        failed = [r for r in results if r["status"] == "failed"]
        print(f"\nBatch complete: {len(ok)} OK, {len(failed)} failed out of {len(results)}")
        if failed:
            sys.exit(1)
    else:
        print("Provide --video for single video or --pattern/--input-dir for batch tracking")
        sys.exit(1)


async def _run_batch(args):
    """Batch process: glob videos, create BatchJob per video, run tracking, log results."""
    import json

    from kawkab.services.batch_service import BatchJob, BatchService, BatchStatus

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    input_dir = Path(args.input_dir)
    videos = sorted(input_dir.glob(args.pattern))
    if not videos:
        print(f"No videos matching '{args.pattern}' in {input_dir}")
        sys.exit(1)

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Use an in-memory BatchService for job tracking (no DB required)
    _ = BatchService()
    job = BatchJob(
        id=0,
        name=f"batch-{args.pattern}-{input_dir.name}",
        status=BatchStatus.RUNNING,
        total_matches=len(videos),
        match_ids=list(range(len(videos))),
        options={"pattern": args.pattern, "input_dir": str(input_dir), "skip": args.skip},
    )

    print(f"Batch processing {len(videos)} videos from {input_dir}")
    print(f"  Pattern: {args.pattern}, Skip: {args.skip}")
    print(f"  Output : {base_out.resolve()}\n")

    for i, video in enumerate(videos):
        video_out_dir = base_out / video.stem
        print(f"[{i + 1}/{len(videos)}] {video.name}...")
        try:
            r = await _process_single_video(str(video), str(video_out_dir), args.skip, args.tracker)
            if r["status"] == "ok":
                job.completed_matches += 1
            else:
                job.failed_matches += 1
        except Exception as e:
            job.failed_matches += 1
            print(f"  Unhandled error: {e}")

        _ = time.time() - 0  # placeholder
        print(
            f"  Progress: {job.completed_matches + job.failed_matches}/{job.total_matches} "
            f"({job.completed_matches} ok, {job.failed_matches} failed)"
        )

    job.status = BatchStatus.COMPLETED if job.failed_matches == 0 else BatchStatus.FAILED
    summary = {
        "name": job.name,
        "total": job.total_matches,
        "completed": job.completed_matches,
        "failed": job.failed_matches,
        "status": job.status.value,
        "options": job.options,
    }
    with open(base_out / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nBatch {'complete' if job.status == BatchStatus.COMPLETED else 'finished with errors'}: "
        f"{job.completed_matches} ok, {job.failed_matches} failed"
    )
    if job.failed_matches:
        sys.exit(1)


def _run_evaluation(args):
    from scripts.evaluate_tracking import main as eval_main

    sys.argv = [
        "evaluate_tracking.py",
        "--tracking",
        args.tracking,
        "--video" if args.video else "",
        args.video if args.video else "",
        "--ground-truth" if args.ground_truth else "",
        args.ground_truth if args.ground_truth else "",
    ]
    eval_main()


def _run_render(args):
    from scripts.render_tracking_overlay import main as render_main

    sys.argv = [
        "render_tracking_overlay.py",
        "--video",
        args.video,
        "--tracking",
        args.tracking,
        "--output",
        args.output,
    ]
    if args.max_frames:
        sys.argv += ["--max-frames", str(args.max_frames)]
    if args.no_ball_trail:
        sys.argv += ["--no-ball-trail"]
    if args.features != "bbox,id,ball":
        sys.argv += ["--features", args.features]
    if args.heatmap_alpha != 0.35:
        sys.argv += ["--heatmap-alpha", str(args.heatmap_alpha)]
    render_main()


def _run_events(args):
    from scripts.detect_events import main as events_main

    sys.argv = ["detect_events.py", "--tracking", args.tracking, "--output", args.output]
    if args.ground_truth:
        sys.argv += ["--ground-truth", args.ground_truth]
    events_main()


def _run_possession(args):
    from kawkab.services.possession_service import PossessionService

    tracking_dir = Path(args.tracking)
    if not tracking_dir.exists():
        print(f"Tracking directory not found: {tracking_dir}")
        sys.exit(1)

    # Load ball tracking data as frame_ball_positions
    ball_path = tracking_dir / "ball_tracking.json"
    frame_ball_positions = []
    if ball_path.exists():
        import json as _json

        with open(ball_path) as f:
            raw = _json.load(f)
        frame_ball_positions = [
            {
                "frame": d["frame"],
                "timestamp_s": d["timestamp"],
                "x": d["x"],
                "y": d["y"],
                "confidence": d.get("conf", 1.0),
            }
            for d in raw
        ]
        print(f"  Loaded {len(frame_ball_positions)} ball tracking frames")

    # Load events
    events = []
    events_path = args.events
    if events_path is None:
        candidates = [
            tracking_dir / "events_detected.json",
            tracking_dir / "events.json",
        ]
        for c in candidates:
            if c.exists():
                events_path = str(c)
                break
    if events_path:
        import json as _json

        with open(events_path) as f:
            events = _json.load(f)
        home_team = args.home
        away_team = args.away
        for ev in events:
            if ev.get("team") is None:
                ev["team"] = ev.get("event_type", "unknown")
        print(f"  Loaded {len(events)} events from {events_path}")
    else:
        home_team = args.home
        away_team = args.away
        print("  No events file provided or found; using ball-tracking heuristics")

    # Run analysis
    svc = PossessionService()
    report = svc.analyze(home_team, away_team, events, frame_ball_positions)

    # Build output
    import json as _json

    chains_data = []
    for i, chain in enumerate(report.home_chains + report.away_chains):
        chains_data.append(
            {
                "chain_index": i,
                "start_time_s": round(chain.start_time_s, 2),
                "end_time_s": round(chain.end_time_s, 2),
                "duration_s": round(chain.duration_s, 2),
                "team": chain.team,
                "player_track_id": chain.player_track_id,
                "player_name": chain.player_name,
                "n_passes": chain.n_passes,
                "ended_by": chain.ended_by,
                "xg_generated": round(chain.xg_generated, 4),
                "is_counter_press": chain.is_counter_press,
            }
        )

    output = {
        "home_possession_pct": report.home_possession_pct,
        "away_possession_pct": report.away_possession_pct,
        "total_chains": len(report.home_chains) + len(report.away_chains),
        "avg_chain_duration_s": report.avg_chain_duration_s,
        "longest_chain_s": report.longest_chain_s,
        "counter_presses": report.counter_presses,
        "notes": report.notes,
        "chains": chains_data,
    }

    output_path = tracking_dir / "possession_chains.json"
    with open(output_path, "w") as f:
        _json.dump(output, f, indent=2)
    print(f"\n  Possession chains saved to {output_path}")
    print(f"  Total chains: {output['total_chains']}")
    print(f"  Avg chain duration: {output['avg_chain_duration_s']}s")
    print(f"  Home possession: {output['home_possession_pct']}%")
    print(f"  Away possession: {output['away_possession_pct']}%")
    print(f"  Counter-presses: {output['counter_presses']}")


def _run_link_players(args):
    import asyncio

    from kawkab.services.storage_service import StorageService

    storage = StorageService()
    storage._get_conn()

    from kawkab.services.cross_match_linking_service import CrossMatchLinkingService

    linker = CrossMatchLinkingService(storage)

    async def run():
        if args.match:
            result = await linker.link_match(args.match)
            print(
                f"  Match {result['match_id']}: linked {result['linked']}, "
                f"flagged {result['flagged_for_review']}"
            )
        else:
            summary = await linker.link_all_matches()
            print(f"  Processed {summary['matches_processed']} matches")
            print(f"  Total linked: {summary['total_linked']}")
            print(f"  Total flagged for review: {summary['total_flagged_for_review']}")
            for mr in summary.get("match_results", []):
                print(
                    f"    Match {mr['match_id']}: linked {mr['linked']}, "
                    f"flagged {mr['flagged_for_review']}"
                )

    asyncio.run(run())


def _run_train_yolo(args):
    """Wire into scripts.fine_tune_yolo training subcommand."""
    from scripts.fine_tune_yolo import prepare_soccer_net_annotations, run_training

    if args.prepare:
        print("Running data preparation first...")
        prepare_soccer_net_annotations(output_dir=Path("data/soccer_net"))

    import argparse as _argparse

    ns = _argparse.Namespace(
        data=args.data,
        model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr=0.001,
        workers=4,
        project="runs/train",
        name="football_finetune",
        export_onnx=False,
    )
    run_training(ns)


def _run_prepare_data(args):
    """Wire into scripts.fine_tune_yolo prepare subcommand."""
    from scripts.fine_tune_yolo import prepare_soccer_net_annotations

    source = Path(args.source) if Path(args.source).exists() else None
    prepare_soccer_net_annotations(output_dir=Path(args.output), source_dir=source)
    print(f"Data preparation complete. Output: {args.output}")


def _run_e2e(args):
    """Run the full analytical E2E pipeline using synthetic data (no DB required)."""
    import asyncio
    import json
    from pathlib import Path

    match_id = args.match_id
    output_dir = args.output

    async def _pipeline():
        nonlocal output_dir
        print(f"E2E Pipeline (synthetic data, match {match_id})")
        print("=" * 60)

        stages = []
        timing = {}

        # Stage 1: Build synthetic track data
        t0 = time.perf_counter()
        import math
        from dataclasses import dataclass, field
        from typing import Any

        @dataclass
        class _Detection:
            bbox: tuple = (0, 0, 10, 10)
            confidence: float = 0.9
            class_id: int = 0
            class_name: str = "person"
            track_id: int | None = None

        @dataclass
        class _FrameDetections:
            frame_number: int = 0
            timestamp: float = 0.0
            detections: list = field(default_factory=list)
            image_width: int = 1280
            image_height: int = 720

        @dataclass
        class _MatchTrackData:
            match_id: int = 0
            fps: float = 30.0
            total_frames: int = 0
            duration_seconds: float = 0.0
            frames: list = field(default_factory=list)
            track_registry: dict = field(default_factory=dict)
            player_teams: dict = field(default_factory=dict)
            tracking_metrics: dict = field(default_factory=dict)
            match_type: str = "test"
            checkpoint_manager: Any = None

        n_frames = 30
        frames = []
        player_teams = {}
        for i in range(11):
            player_teams[i + 1] = "home"
        for i in range(11):
            player_teams[100 + i] = "away"

        for fno in range(n_frames):
            ts = fno / 30.0
            dets = [
                _Detection(
                    bbox=(640 - 5, 360 - 5, 640 + 5, 360 + 5),
                    confidence=0.95,
                    class_id=32,
                    class_name="sports ball",
                    track_id=999,
                )
            ]
            for j in range(11):
                px = 200 + j * 60 + 10 * math.sin(ts + j)
                py = 50 + j * 55 + 10 * math.cos(ts * 0.5 + j)
                dets.append(
                    _Detection(
                        bbox=(px - 15, py - 15, px + 15, py + 15),
                        confidence=0.9,
                        class_id=0,
                        class_name="person",
                        track_id=j + 1,
                    )
                )
            for j in range(11):
                px = 800 + j * 40 + 10 * math.sin(ts + j * 0.7)
                py = 50 + j * 55 + 10 * math.cos(ts * 0.4 + j * 0.5)
                dets.append(
                    _Detection(
                        bbox=(px - 15, py - 15, px + 15, py + 15),
                        confidence=0.9,
                        class_id=0,
                        class_name="person",
                        track_id=100 + j,
                    )
                )
            frames.append(
                _FrameDetections(
                    frame_number=fno,
                    timestamp=ts,
                    detections=dets,
                    image_width=1280,
                    image_height=720,
                )
            )

        track_registry = {}
        for tid in list(player_teams.keys()):
            track_registry[tid] = {"first_pixel_x": 200.0 if tid <= 11 else 800.0}

        _ = _MatchTrackData(
            match_id=match_id,
            fps=30.0,
            total_frames=n_frames,
            duration_seconds=n_frames / 30.0,
            frames=frames,
            track_registry=track_registry,
            player_teams=player_teams,
            tracking_metrics={},
            match_type="e2e_test",
        )

        timing["build_synthetic_data"] = time.perf_counter() - t0
        stages.append(
            {
                "stage": "build_synthetic_data",
                "status": "OK",
                "time_s": round(timing["build_synthetic_data"], 3),
            }
        )
        print(
            f"  [OK] Built synthetic track data ({n_frames} frames, 22 players)"
            f"  ({timing['build_synthetic_data']:.2f}s)"
        )

        # Stage 2: xG from synthetic events
        t0 = time.perf_counter()
        from kawkab.core.xg_model import compute_xg

        xg_vals = [compute_xg(8 + i * 5, 15 + i * 10) for i in range(4)]
        timing["xg_computation"] = time.perf_counter() - t0
        stages.append(
            {
                "stage": "xg_computation",
                "status": "OK",
                "values": [round(v, 3) for v in xg_vals],
                "time_s": round(timing["xg_computation"], 3),
            }
        )
        print(
            f"  [OK] xG computation: {[round(v, 3) for v in xg_vals]}"
            f"  ({timing['xg_computation']:.2f}s)"
        )

        # Stage 3: xT model
        t0 = time.perf_counter()
        from kawkab.core.xt_model import ExpectedThreatModel

        xt_events = [
            {
                "type": "pass",
                "team": "home" if i % 2 == 0 else "away",
                "completed": True,
                "timestamp": float(i),
                "start_x": float(10 + (i % 80)),
                "start_y": float(10 + (i % 50)),
                "end_x": float(20 + (i % 70)),
                "end_y": float(10 + (i % 50)),
            }
            for i in range(50)
        ]
        xt_model = ExpectedThreatModel()
        xt_model.build_transition_matrix(xt_events)
        timing["xt_model"] = time.perf_counter() - t0
        stages.append({"stage": "xt_model", "status": "OK", "time_s": round(timing["xt_model"], 3)})
        print(f"  [OK] xT model built ({len(xt_events)} events)  ({timing['xt_model']:.2f}s)")

        # Stage 4: VAEP
        t0 = time.perf_counter()
        from kawkab.core.vaep import compute_vaep

        vaep_events = []
        for i in range(20):
            team = "home" if i % 2 == 0 else "away"
            if i % 4 == 0:
                vaep_events.append(
                    {
                        "type": "shot",
                        "team": team,
                        "timestamp": float(i),
                        "x": 50 + i,
                        "y": 34 + i,
                        "is_goal": i % 8 == 0,
                        "xg": 0.1,
                    }
                )
            elif i % 4 == 1:
                vaep_events.append(
                    {
                        "type": "pass",
                        "team": team,
                        "timestamp": float(i),
                        "x": 30 + i,
                        "y": 20 + i,
                        "completed": True,
                    }
                )
            elif i % 4 == 2:
                vaep_events.append(
                    {
                        "type": "tackle",
                        "team": team,
                        "timestamp": float(i),
                        "x": 40 + i,
                        "y": 30 + i,
                    }
                )
            else:
                vaep_events.append(
                    {"type": "carry", "team": team, "timestamp": float(i), "x": 50 + i, "y": 34 + i}
                )
        vaep_result = compute_vaep(vaep_events)
        timing["vaep"] = time.perf_counter() - t0
        stages.append(
            {
                "stage": "vaep",
                "status": "OK",
                "count": len(vaep_result),
                "time_s": round(timing["vaep"], 3),
            }
        )
        print(f"  [OK] VAEP computed ({len(vaep_result)} values)  ({timing['vaep']:.2f}s)")

        # Stage 5: Pitch control
        t0 = time.perf_counter()
        from kawkab.core.pitch_control import VoronoiPitchControl

        pc = VoronoiPitchControl()
        home_pos = [(float(20 + i * 6), float(10 + i * 5)) for i in range(11)]
        away_pos = [(float(50 + i * 5), float(30 + i * 3)) for i in range(11)]
        pc_frame = pc.compute_frame_control(home_pos, away_pos, ball_pos=(50, 34))
        timing["pitch_control"] = time.perf_counter() - t0
        stages.append(
            {
                "stage": "pitch_control",
                "status": "OK",
                "home_pct": round(pc_frame.home_control_pct, 1),
                "away_pct": round(pc_frame.away_control_pct, 1),
                "time_s": round(timing["pitch_control"], 3),
            }
        )
        print(
            f"  [OK] Pitch control: home={pc_frame.home_control_pct:.1f}% away={pc_frame.away_control_pct:.1f}%"
            f"  ({timing['pitch_control']:.2f}s)"
        )

        # Stage 6: Formation analysis
        t0 = time.perf_counter()
        from kawkab.core.formation_analysis import FormationAnalyzer

        fa = FormationAnalyzer()
        positions = [(20 + i * 8, 15 + i * 4) for i in range(10)]
        formation = fa._classify_formation(positions)
        timing["formation_analysis"] = time.perf_counter() - t0
        stages.append(
            {
                "stage": "formation_analysis",
                "status": "OK",
                "formation": formation,
                "time_s": round(timing["formation_analysis"], 3),
            }
        )
        print(f"  [OK] Formation analysis: {formation}  ({timing['formation_analysis']:.2f}s)")

        # Stage 7: Win probability
        t0 = time.perf_counter()
        from kawkab.core.win_probability import compute_win_probability

        wp_events = [
            {
                "type": "shot",
                "team": "home" if i % 3 == 0 else "away",
                "timestamp": float(i * 60),
                "xg": 0.1,
                "is_goal": i % 5 == 0,
            }
            for i in range(10)
        ]
        wp = compute_win_probability(wp_events)
        timing["win_probability"] = time.perf_counter() - t0
        stages.append(
            {
                "stage": "win_probability",
                "status": "OK",
                "home_win": round(wp.starting_home_win, 3),
                "away_win": round(wp.starting_away_win, 3),
                "draw": round(wp.starting_draw, 3),
                "time_s": round(timing["win_probability"], 3),
            }
        )
        print(
            f"  [OK] Win probability: home={wp.starting_home_win:.1%} draw={wp.starting_draw:.1%} away={wp.starting_away_win:.1%}"
            f"  ({timing['win_probability']:.2f}s)"
        )

        # Stage 8: Momentum
        t0 = time.perf_counter()
        from kawkab.core.momentum import compute_momentum_index

        mom_events = [
            {
                "timestamp": float(i),
                "type": "shot" if i % 5 == 0 else "pass",
                "team": "home" if i % 2 == 0 else "away",
                "x": float(50 + (i % 50)),
                "y": float(34 + (i % 30)),
                "xg": 0.1,
                "is_goal": i % 10 == 0,
                "completed": True,
            }
            for i in range(50)
        ]
        momentum = compute_momentum_index(mom_events)
        timing["momentum"] = time.perf_counter() - t0
        stages.append(
            {
                "stage": "momentum",
                "status": "OK",
                "home_pct": round(momentum.home_momentum_pct, 1),
                "away_pct": round(momentum.away_momentum_pct, 1),
                "time_s": round(timing["momentum"], 3),
            }
        )
        print(
            f"  [OK] Momentum: home={momentum.home_momentum_pct:.1f}% away={momentum.away_momentum_pct:.1f}%"
            f"  ({timing['momentum']:.2f}s)"
        )

        # Summary
        total_time = sum(s["time_s"] for s in stages)
        all_ok = all("FAILED" not in s.get("status", "") for s in stages)
        print()
        print("=" * 60)
        print(f"E2E Pipeline {'PASSED' if all_ok else 'FAILED'}")
        print(f"  Total time: {total_time:.2f}s")
        for s in stages:
            sc = "OK" if "FAILED" not in s.get("status", "") else "FAIL"
            print(f"  [{sc}] {s['stage']}: {s.get('time_s', 0):.2f}s")

        report = {
            "match_id": match_id,
            "status": "PASSED" if all_ok else "FAILED",
            "total_time_s": round(total_time, 3),
            "stages": stages,
        }

        if output_dir:
            out_path = Path(output_dir) / f"e2e_report_{match_id}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2))
            print(f"\n  Report saved to {out_path}")

        return all_ok

    success = asyncio.run(_pipeline())
    sys.exit(0 if success else 1)


def _run_validate(args):
    """Build the reproducible model-validation report (Phase 2 trust layer).

    Writes validation_report.json + validation_report.md into --out.
    Exit code 0 = report written (even if numbers are bad — honesty is
    the gate, not optimism), matching scripts/validate_models.py.
    """

    from kawkab.services.validation_report_service import ValidationReportService

    svc = ValidationReportService(corpus_dir=args.corpus)
    report = svc.write_report(out_dir=args.out, max_matches=args.max_matches)

    print("Validation report written")
    print(f"  Corpus : {report['corpus_dir']}")
    print(f"  Output : {Path(args.out) / 'validation_report.json'}")
    summary = report.get("summary", {})
    print(
        f"  Sections evaluated: {summary.get('sections_evaluated', 0)} "
        f"(skipped: {summary.get('sections_skipped', 0)})"
    )
    print(f"  Model cards registered: {summary.get('model_cards_registered', 0)}")
    for s in report.get("sections", []):
        status = s.get("status", "?")
        if status == "evaluated":
            keys = [k for k in s.get("metrics", {}) if k != "cards"]
            print(f"  [{status.upper()}] {s['name']}: {', '.join(keys) or 'ok'}")
        else:
            print(f"  [{status.upper()}] {s['name']}: {s.get('reason', '')}")


def _run_import(args):
    """Vendor ingest CLI: one file, or a whole directory as one season.

    Directory mode deduplicates via the migration-031 external-ID
    registry — re-running the same directory imports 0 new matches.
    Exit code 0 when nothing failed, 1 when at least one file failed.
    """
    import asyncio

    from kawkab.services.season_import_service import SeasonImportService
    from kawkab.services.storage_service import StorageService

    storage = StorageService()
    asyncio.run(storage.initialize())  # runs migrations (incl. 031) + connects

    path = Path(args.path)
    svc = SeasonImportService(storage)

    async def run():
        if path.is_dir():
            return await svc.import_statsbomb_directory(
                path,
                competition=args.competition,
                season_id=args.season_id,
                match_date=args.match_date,
                max_matches=args.max_matches,
            )
        return await svc.import_single_file(
            path,
            competition=args.competition,
            season_id=args.season_id,
            match_date=args.match_date,
        )

    summary = asyncio.run(run())

    if path.is_dir():
        print(f"Season import: {summary['directory']}")
        print(
            f"  Files scanned: {summary['total_files']} "
            f"(new event files this run: {summary['eligible']})"
        )
        print(
            f"  Imported: {summary['imported']}  "
            f"Already present: {summary['skipped_already']}  "
            f"Not event files: {summary['skipped_not_events']}  "
            f"Failed: {summary['failed']}"
        )
        for m in summary["matches"]:
            if m["status"] == "failed":
                print(f"    FAILED {m['file']}: {m.get('error', '')}")
            elif m["status"] == "imported":
                print(
                    f"    imported {m['file']} -> match {m['match_id']} ({m.get('match_name', '')})"
                )
        raise SystemExit(1 if summary["failed"] else 0)

    print(f"{summary['status']}: {summary['file']} -> match {summary.get('match_id')}")
    raise SystemExit(0 if summary["status"] != "failed" else 1)


def _run_benchmark(args):
    """Run analytical module benchmarks."""
    import json

    module = args.module
    n_iterations = args.iterations
    output_path = args.output

    from tests.unit.test_performance_benchmarks import (
        MODULES_TO_BENCHMARK,
        N_EVENTS_DEFAULT,
        BenchmarkRunner,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tests"))

    runner = BenchmarkRunner(n_iterations=n_iterations)

    modules_to_run = [module] if module else MODULES_TO_BENCHMARK

    print(f"Benchmarking {len(modules_to_run)} module(s) ({n_iterations} iterations each)")
    print("=" * 60)

    for mod in modules_to_run:
        n_events = N_EVENTS_DEFAULT.get(mod, 100)
        print(f"  Running {mod} ({n_events} events)... ", end="")
        try:
            result = runner.run_benchmark(mod, n_events)
            status = "PASS"
            threshold = runner.thresholds.get(mod, float("inf"))
            if result.mean_ms >= threshold:
                status = "WARN"
            print(
                f"{status}  mean={result.mean_ms:.1f}ms  "
                f"min={result.min_ms:.1f}ms  max={result.max_ms:.1f}ms  "
                f"p95={result.p95_ms:.1f}ms  (threshold={threshold:.0f}ms)"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")

    # Summary
    print()
    print("=" * 60)
    checks = runner.check_thresholds()
    passed = sum(1 for _, _, _, s in checks if s == "PASS")
    warned = sum(1 for _, _, _, s in checks if s == "WARN")
    print(f"Results: {passed} passed, {warned} warnings (soft) out of {len(checks)} modules")

    report = runner.report()
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"  Report saved to {output_path}")


if __name__ == "__main__":
    main()
