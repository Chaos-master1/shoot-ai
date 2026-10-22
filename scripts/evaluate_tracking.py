"""Evaluate tracking quality against ground truth data.

Supports:
  - Metrica Sports sample data (XY tracking, 2 matches)
  - Self-evaluation (track fragmentation, team accuracy, ID stability)

Usage:
  # Self-evaluation on our tracking output
  python scripts/evaluate_tracking.py --self tracking_output/track_summary.json

  # Against Metrica ground truth
  python scripts/evaluate_tracking.py --metrica data/ground_truth/metrica
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("evaluate_tracking")

# ── Data Models ────────────────────────────────────────────────────


@dataclass
class TrackFrame:
    frame: int
    timestamp: float
    x: float
    y: float


@dataclass
class GroundTruthTrack:
    player_id: str  # jersey number or team_role
    frames: list[TrackFrame] = field(default_factory=list)


@dataclass
class GroundTruthMatch:
    """Parsed ground truth tracking data from Metrica."""

    home_tracks: dict[str, GroundTruthTrack] = field(default_factory=dict)
    away_tracks: dict[str, GroundTruthTrack] = field(default_factory=dict)
    ball_track: GroundTruthTrack = field(default_factory=lambda: GroundTruthTrack("ball"))
    fps: float = 25.0
    total_frames: int = 0


# ── Ground Truth Parsers ────────────────────────────────────────────


def parse_metrica_csv(csv_path: Path, team_label: str) -> dict[str, GroundTruthTrack]:
    """Parse a Metrica tracking CSV into per-player tracks.

    Metrica format (after 3 header rows):
      Period, Frame, Time[s], PlayerXX_x, PlayerXX_y, ..., Ball_x, Ball_y
    Coordinates are normalized 0-1 (pitch-relative).
    """
    # Read all lines and parse
    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 4:
        return {}

    header1 = lines[0].strip().split(",")
    header2 = lines[1].strip().split(",")
    data_rows = lines[3:]  # skip 3 header rows

    # Build column index: subtract 3 header cols (Period, Frame, Time), rest are (x,y) pairs
    n_players = (len(header1) - 4) // 2
    player_nums = []
    for i in range(n_players):
        col_x = 3 + i * 2
        num_str = header2[col_x] if col_x < len(header2) else ""
        if num_str.strip():
            player_nums.append(num_str.strip())
        else:
            player_nums.append(f"p{i}")

    tracks: dict[str, GroundTruthTrack] = {}
    for pn in player_nums:
        tracks[pn] = GroundTruthTrack(pn)

    for row in data_rows:
        parts = row.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            frame = int(parts[1])
            ts = float(parts[2])
        except (ValueError, IndexError):
            continue
        for i, pn in enumerate(player_nums):
            col_x = 3 + i * 2
            col_y = col_x + 1
            try:
                vx = float(parts[col_x])
                vy = float(parts[col_y])
                tracks[pn].frames.append(TrackFrame(frame, ts, vx, vy))
            except (ValueError, IndexError):
                pass

    return tracks


def load_metrica_ground_truth(data_dir: Path) -> GroundTruthMatch:
    """Load both home and away Metrica CSVs."""
    gt = GroundTruthMatch()
    home_csv = list(data_dir.rglob("*RawTrackingData_Home_Team.csv"))
    away_csv = list(data_dir.rglob("*RawTrackingData_Away_Team.csv"))

    if home_csv:
        gt.home_tracks = parse_metrica_csv(home_csv[0], "home")
    if away_csv:
        gt.away_tracks = parse_metrica_csv(away_csv[0], "away")

    if home_csv:
        with open(home_csv[0], encoding="utf-8") as f:
            lines = f.readlines()
            data_end = len(lines) - 3  # after 3 header rows
        # Estimate total frames and fps from last row
        if data_end > 0:
            last_row = lines[-1].strip().split(",")
            if len(last_row) >= 3:
                try:
                    gt.total_frames = int(last_row[1])
                    last_time = float(last_row[2])
                    if gt.total_frames > 1 and last_time > 0:
                        gt.fps = gt.total_frames / last_time
                except (ValueError, IndexError):
                    pass
            gt.total_frames = max(gt.total_frames, data_end)
            for val in gt.home_tracks.values():
                if val.frames:
                    gt.total_frames = max(gt.total_frames, val.frames[-1].frame)
                    break

    logger.info(
        f"Loaded Metrica ground truth: "
        f"{len(gt.home_tracks)} home, {len(gt.away_tracks)} away tracks, "
        f"~{gt.total_frames} frames"
    )
    return gt


def load_self_tracking(tracking_dir: Path) -> dict[str, Any]:
    """Load our own tracking output for self-evaluation."""
    summary_path = tracking_dir / "track_summary.json"
    frames_path = tracking_dir / "frames_compact.pkl"

    result = {}
    if summary_path.exists():
        with open(summary_path) as f:
            result["summary"] = json.load(f)
        logger.info(f"Loaded tracking summary: {len(result['summary'].get('tracks', {}))} tracks")

    if frames_path.exists():
        import pickle

        with open(frames_path, "rb") as f:
            result["frames"] = pickle.load(f)
        logger.info(f"Loaded {len(result.get('frames', []))} sampled frames")

    return result


# ── Metrics ────────────────────────────────────────────────────────


def compute_fragmentation(track_summary: dict) -> dict:
    """Compute track fragmentation metrics from summary."""
    tracks = track_summary.get("tracks", {})
    if not tracks:
        return {"error": "no tracks"}

    lifetimes = [t["frames"] for t in tracks.values()]
    pcts = [t["pct"] for t in tracks.values()]

    total_frames = sum(lifetimes)
    n_tracks = len(tracks)
    avg_lifetime = total_frames / n_tracks if n_tracks > 0 else 0

    quality_buckets = {"high (>10%)": 0, "medium (2-10%)": 0, "low (<2%)": 0}
    for p in pcts:
        if p > 10:
            quality_buckets["high (>10%)"] += 1
        elif p > 2:
            quality_buckets["medium (2-10%)"] += 1
        else:
            quality_buckets["low (<2%)"] += 1

    team_dist: dict[str, int] = {}
    for t in tracks.values():
        team = t.get("team", "?")
        team_dist[team] = team_dist.get(team, 0) + 1

    return {
        "n_tracks": n_tracks,
        "total_tracked_frames": total_frames,
        "avg_lifetime_frames": round(avg_lifetime, 1),
        "quality_buckets": quality_buckets,
        "team_distribution": team_dist,
    }


def compute_id_stability(tracking_data: dict) -> dict:
    """Estimate ID switch rate from sampled frame data."""
    frames = tracking_data.get("frames", [])
    if not frames:
        return {"error": "no frame data"}

    # Track ID appearance continuity across time windows
    prev_ids: set[int] = set()
    total_switches = 0
    total_appearances = 0

    for fd in frames:
        current_ids = set()
        for det in fd.get("detections", []):
            tid = det[3] if len(det) > 3 else None
            if tid is not None:
                current_ids.add(tid)

        disappeared = prev_ids - current_ids
        appeared = current_ids - prev_ids

        total_switches += len(disappeared) + len(appeared)
        total_appearances += len(current_ids) + len(prev_ids)
        prev_ids = current_ids

    switch_rate = total_switches / max(total_appearances, 1)
    return {
        "total_id_switches": total_switches,
        "total_appearances": total_appearances,
        "switch_rate": round(switch_rate, 4),
    }


def compare_to_ground_truth(
    predicted_dir: Path,
    gt_match: GroundTruthMatch,
) -> dict:
    """Compare our tracking output against Metrica ground truth.

    Note: This aligns by frame number. Our tracking uses pixel coords,
    Metrica uses normalized pitch coords. Direct comparison requires
    homography projection from pixel → pitch coordinates.
    """
    # For now, compute coverage statistics
    gt_total_players = len(gt_match.home_tracks) + len(gt_match.away_tracks)
    gt_total_player_frames = sum(
        len(t.frames)
        for t in list(gt_match.home_tracks.values()) + list(gt_match.away_tracks.values())
    )

    return {
        "ground_truth_players": gt_total_players,
        "ground_truth_player_frames": gt_total_player_frames,
        "ground_truth_fps": gt_match.fps,
        "ground_truth_total_frames": gt_match.total_frames,
        "note": "Full pixel-to-pitch comparison requires homography projection (see evaluate_pitch_alignment)",
    }


def evaluate_pitch_alignment(gt_match: GroundTruthMatch) -> dict:
    """Evaluate pitch coverage and player density from ground truth."""
    x_coords = []
    y_coords = []
    for track in list(gt_match.home_tracks.values()) + list(gt_match.away_tracks.values()):
        for f in track.frames[:1000]:  # sample
            x_coords.append(f.x)
            y_coords.append(f.y)

    if not x_coords:
        return {"error": "no coordinates"}

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return {
        "pitch_x_range": [round(x_min, 3), round(x_max, 3)],
        "pitch_y_range": [round(y_min, 3), round(y_max, 3)],
        "coverage_area_pct": round((x_max - x_min) * (y_max - y_min) * 100, 1),
    }


# ── Reporter ────────────────────────────────────────────────────────


def print_report(results: dict[str, Any]):
    """Print a formatted evaluation report."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  TRACKING EVALUATION REPORT")
    print(f"{sep}")

    for section, data in results.items():
        if isinstance(data, dict) and "error" in data:
            print(f"\n  [{section}] ERROR: {data['error']}")
            continue

        print(f"\n  --- {section.upper()} ---")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for sk, sv in v.items():
                        print(f"      {sk}: {sv}")
                elif isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"    {data}")
    print(f"{sep}\n")


# ── Main ────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tracking quality")
    parser.add_argument("--self", type=str, help="Self-evaluate tracking_output/ directory")
    parser.add_argument("--metrica", type=str, help="Metrica ground truth data directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ground_truth/metrica",
        help="Metrica data directory (default: data/ground_truth/metrica)",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}

    # Self evaluation
    if args.self:
        tracking_dir = Path(args.self)
        if not tracking_dir.exists():
            logger.error(f"Tracking directory not found: {args.self}")
            sys.exit(1)
        data = load_self_tracking(tracking_dir)
        if "summary" in data:
            results["fragmentation"] = compute_fragmentation(data["summary"])
        if "frames" in data:
            results["id_stability"] = compute_id_stability(data)

    # Metrica ground truth evaluation
    metrica_dir = Path(args.metrica) if args.metrica else Path("data/ground_truth/metrica")
    if metrica_dir.exists():
        gt = load_metrica_ground_truth(metrica_dir)
        results["ground_truth"] = {
            "home_players": len(gt.home_tracks),
            "away_players": len(gt.away_tracks),
            "total_players": len(gt.home_tracks) + len(gt.away_tracks),
            "fps": gt.fps,
            "total_frames": gt.total_frames,
        }
        results["pitch_alignment"] = evaluate_pitch_alignment(gt)
        if args.self:
            results["vs_ground_truth"] = compare_to_ground_truth(Path(args.self), gt)
    else:
        logger.info(f"No Metrica data at {metrica_dir}, skipping ground truth comparison")

    print_report(results)


if __name__ == "__main__":
    main()
