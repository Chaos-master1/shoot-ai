"""Hot-path micro-benchmarks for the analysis pipeline.

Run directly (never pytest-collected):

    .venv-linux/bin/python scripts/bench_hot_path.py

Measures the per-call latency of the four hot paths that run over every
frame of a tracked match, on deterministic synthetic data shaped like a
real 90-minute 25fps track (135k frames would take minutes to build, so
we scale to a 60s window and report per-frame costs that extrapolate
linearly):

  1. AnalysisHandler._compute_overlay_data  — run once per analysis,
     consumed by get_overlay_data for the animated pitch view
  2. json.dumps of that overlay payload      — the serialization that
     crosses the Qt bridge on every UI poll
  3. ProAnalyticsHandler._frames_to_obv_schema — per OBV/off-ball report
  4. _compute_hot_zones                      — per live-tagging pitch map

Reports best/median wall time and the cProfile top-N for the two
heaviest paths.
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData  # noqa: E402
from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler  # noqa: E402
from kawkab.ui.bridge_handlers.bridge_live import _compute_hot_zones  # noqa: E402
from kawkab.ui.bridge_handlers.bridge_pro_analytics import ProAnalyticsHandler  # noqa: E402

FPS = 25.0
SECONDS = 60
N_FRAMES = int(FPS * SECONDS)
N_PLAYERS = 22
REPEATS = 7


def build_track_data() -> MatchTrackData:
    """Deterministic synthetic track: 22 tracked players + ball per frame."""
    frames: list[FrameDetections] = []
    for i in range(N_FRAMES):
        dets: list[Detection] = []
        for p in range(N_PLAYERS):
            # Players drift sinusoidally; formulas keep the run deterministic.
            x = 100 + 800 * (0.5 + 0.5 * ((p * 0.7 + i * 0.02) % 1.0))
            y = 50 + 500 * (0.5 + 0.5 * ((p * 1.3 + i * 0.03) % 1.0))
            dets.append(
                Detection(
                    bbox=(x, y, x + 30.0, y + 70.0),
                    confidence=0.9,
                    class_id=0,
                    class_name="person",
                    track_id=p,
                )
            )
        bx = 400 + 300 * ((i * 0.05) % 1.0)
        by = 300 + 200 * ((i * 0.11) % 1.0)
        dets.append(
            Detection(
                bbox=(bx, by, bx + 22.0, by + 22.0),
                confidence=0.95,
                class_id=32,
                class_name="sports ball",
                track_id=None,
            )
        )
        frames.append(
            FrameDetections(
                frame_number=i,
                timestamp=i / FPS,
                detections=dets,
                image_width=1920,
                image_height=1080,
            )
        )
    return MatchTrackData(
        match_id=1,
        fps=FPS,
        total_frames=N_FRAMES,
        duration_seconds=SECONDS,
        frames=frames,
        track_registry={p: {"first_seen": 0} for p in range(N_PLAYERS)},
        player_teams={p: ("home" if p % 2 == 0 else "away") for p in range(N_PLAYERS)},
    )


def build_obv_frames(track: MatchTrackData) -> list[dict]:
    out: list[dict] = []
    for f in track.frames:
        out.append(
            {
                "timestamp": f.timestamp,
                "player_detections": [
                    {"bbox": [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]]}
                    for d in f.detections
                    if d.class_name == "person"
                ],
                "ball_detections": [
                    {"bbox": [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]]}
                    for d in f.detections
                    if d.class_name == "sports ball"
                ],
            }
        )
    return out


def bench(label: str, fn, repeats: int = REPEATS) -> float:
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    med = statistics.median(times)
    per_frame_us = best / N_FRAMES * 1e6
    print(
        f"{label:<46} best={best * 1e3:8.2f} ms  median={med * 1e3:8.2f} ms"
        f"  ({per_frame_us:6.1f} µs/frame)"
    )
    return result


def profile_top(label: str, fn, top: int = 8) -> None:
    prof = cProfile.Profile()
    prof.enable()
    fn()
    prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
    stats.print_stats(top)
    print(f"\n--- cProfile top {top}: {label} ---")
    lines = buf.getvalue().splitlines()
    for line in lines[4 : 4 + top + 4]:
        print(line)
    print()


def main() -> int:
    print(f"synthetic track: {N_FRAMES} frames @ {FPS:.0f} fps, {N_PLAYERS} players + ball\n")
    track = build_track_data()
    analysis = AnalysisHandler(bridge=None, services={}, rate_limiter=None)
    pro = ProAnalyticsHandler(bridge=None, services={}, rate_limiter=None)

    overlay = bench("1 overlay_data (handler._compute_overlay_data)", lambda: analysis._compute_overlay_data(track))
    payload = json.dumps(overlay, separators=(",", ":"))
    bench("2 json.dumps(overlay) — bridge crossing", lambda: json.dumps(overlay, separators=(",", ":")))
    obv_frames = build_obv_frames(track)
    bench("3 frames_to_obv_schema (pro handler)", lambda: pro._frames_to_obv_schema(obv_frames))
    events = [e for f in overlay[:900] for e in f["p"]]
    bench("4 _compute_hot_zones (live pitch map)", lambda: _compute_hot_zones(events))
    print(f"(overlay payload size: {len(payload) / 1024:.0f} KiB for {SECONDS}s window)")

    profile_top("overlay_data", lambda: analysis._compute_overlay_data(track))
    profile_top("frames_to_obv_schema", lambda: pro._frames_to_obv_schema(obv_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
