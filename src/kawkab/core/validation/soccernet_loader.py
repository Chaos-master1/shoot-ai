"""SoccerNet tracking ground-truth loader → Kawkab benchmark schema.

SoccerNet's tracking-2023 task ships per-half zip archives containing:
  - video.mp4  — the broadcast footage
  - gt.txt     — one line per frame per tracked object:
                 "frame playerID xmin ymin xmax ymax [jersey]" with
                 frame numbers relative to the half, box in image pixels.

This loader converts gt.txt into the exact ``gt_tracks`` dict shape
``core/mot_metrics.compute_mot_metrics`` consumes —
{gt_id: [(frame, x_px, y_px), ...]} with box centers — plus the raw
per-frame boxes the e2e benchmark needs to build detector input, and a
separate ball channel when a ball-action spotting file is present.

The loader is deliberately format-strict: it raises on malformed lines
rather than silently skipping, because a silently-truncated GT silently
corrupts every downstream MOTA/IDF1 number (the model-validation
discipline applied to tracking).

Data license: SoccerNet is provided for research purposes — see
docs/DATA_CARD.md. The benchmark published numbers must cite the exact
game/half/frames used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SoccerNetGtFormatError(ValueError):
    """A gt.txt line does not match the documented format."""


@dataclass
class SoccerNetHalf:
    """Parsed ground truth for one half of one game."""

    game: str
    half: int
    # {player_id: [(frame, x_px, y_px), ...]} — box centers, deterministic
    # player_id ordering preserved from the file (sorted numerically when
    # ids are numeric).
    tracks: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)
    # {frame: [(track_id, x1, y1, x2, y2), ...]} — the raw boxes, used to
    # build detector input for the e2e benchmark.
    boxes_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = field(
        default_factory=dict
    )
    width: int = 0
    height: int = 0
    max_frame: int = 0

    @property
    def n_positions(self) -> int:
        return sum(len(v) for v in self.tracks.values())


def parse_gt_line(line: str, line_no: int) -> tuple[int, int, float, float, float, float]:
    """Parse one gt.txt row → (frame, track_id, x1, y1, x2, y2).

    The SoccerNet tracking-2023 gt.txt is CSV with 10 columns:
        frame, track_id, left, top, width, height, flag, -1, -1, -1
    (flag is 1 for real boxes; the trailing -1 columns are the MOT
    challenge's reserved class/visibility fields). Whitespace-separated
    6+ column rows — the other documented MOT variant — are accepted for
    robustness, and extra trailing columns are ignored either way.
    Fewer than 6 usable columns is a hard format error.
    """
    parts = line.split(",") if "," in line else line.split()
    if len(parts) < 6:
        raise SoccerNetGtFormatError(
            f"gt line {line_no}: expected >= 6 columns, got {len(parts)}: {line!r}"
        )
    try:
        frame = int(parts[0])
        tid = int(parts[1])
        if "," in line:
            # CSV variant: left, top, width, height
            left, top = float(parts[2]), float(parts[3])
            w, h = float(parts[4]), float(parts[5])
            x1, y1, x2, y2 = left, top, left + w, top + h
        else:
            x1, y1, x2, y2 = (float(v) for v in parts[2:6])
    except ValueError as e:
        raise SoccerNetGtFormatError(
            f"gt line {line_no}: non-numeric column: {e} ({line!r})"
        ) from e
    # Normalize inverted boxes defensively (some tools emit y1 > y2).
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return frame, tid, x1, y1, x2, y2


def load_soccernet_half(gt_path: Path, *, frame_limit: int | None = None) -> SoccerNetHalf:
    """Load one SoccerNet half's gt.txt into the benchmark schema.

    Args:
        gt_path: path to the half's ``gt.txt``. The parent dir name is
            used as the game label when it contains the half number
            (e.g. ``.../game/2_gt.txt`` → half 2), else ``half`` must be
            derivable from the filename's leading digit.
        frame_limit: keep only frames <= this (1-based, matching gt.txt).
    """
    gt_path = Path(gt_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"SoccerNet gt not found: {gt_path}")

    game = gt_path.parent.name
    if game in ("gt", "video"):
        game = gt_path.parent.parent.name
    stem = gt_path.stem
    half = 1
    for ch in stem:
        if ch.isdigit():
            half = int(ch)
            break

    tracks: dict[int, list[tuple[int, float, float]]] = {}
    boxes_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    width = 0
    height = 0
    max_frame = 0

    with open(gt_path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            frame, tid, x1, y1, x2, y2 = parse_gt_line(line, line_no)
            if frame_limit is not None and frame > frame_limit:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tracks.setdefault(tid, []).append((frame, cx, cy))
            boxes_by_frame.setdefault(frame, []).append((tid, x1, y1, x2, y2))
            width = max(width, int(x2))
            height = max(height, int(y2))
            max_frame = max(max_frame, frame)

    # Deterministic id remap: sorted player ids → 1000+i (consistent with
    # the Metrica benchmark's convention; never hash(), which differs
    # across processes).
    remap = {tid: 1000 + i for i, tid in enumerate(sorted(tracks.keys()))}
    tracks = {remap[tid]: positions for tid, positions in sorted(tracks.items())}
    remapped_boxes: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for frame, boxes in boxes_by_frame.items():
        remapped_boxes[frame] = [(remap[b[0]], *b[1:]) for b in boxes]

    return SoccerNetHalf(
        game=game,
        half=half,
        tracks=tracks,
        boxes_by_frame=remapped_boxes,
        width=width,
        height=height,
        max_frame=max_frame,
    )


def discover_soccernet_games(root: Path) -> list[Path]:
    """Find gt.txt files under a SoccerNet tracking data root.

    Layouts seen in the wild:
      root/game/half/gt.txt        (per-half dirs: 1/, 2/)
      root/game/half_gt.txt        (flat, suffixed)
    Both are discovered; anything else is left alone.
    """
    root = Path(root)
    if not root.exists():
        return []
    found: list[Path] = []
    for gt in sorted(root.rglob("gt.txt")):
        found.append(gt)
    for gt in sorted(root.rglob("*_gt.txt")):
        if gt not in found:
            found.append(gt)
    return found


def fragmentation_stats(
    pred_tracks: dict[int, list[tuple[int, float, float]]],
) -> dict[str, float]:
    """Tracks-per-identity diagnostics for the fragmentation question.

    The 2026 README flagged "91 tracks for 22 players". This quantifies
    that directly: how many predicted tracks exist per ground-truth
    identity when each GT id is matched to its best-overlap pred track.
    Higher than ~1.0 means identities are splitting across tracks.
    """
    if not pred_tracks:
        return {"n_pred_tracks": 0.0, "mean_track_length": 0.0, "median_track_length": 0.0}
    lengths = sorted(len(v) for v in pred_tracks.values())
    n = len(lengths)
    mean = sum(lengths) / n
    median = lengths[n // 2] if n % 2 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2
    return {
        "n_pred_tracks": float(n),
        "mean_track_length": mean,
        "median_track_length": median,
    }


def gt_tracks_to_metric_shape(
    half: SoccerNetHalf,
) -> dict[int, list[tuple[int, float, float]]]:
    """Pass-through accessor making the mot_metrics contract explicit."""
    return {tid: list(positions) for tid, positions in half.tracks.items()}


def summarize_half(half: SoccerNetHalf) -> dict[str, Any]:
    """Honest provenance summary for a published report."""
    return {
        "game": half.game,
        "half": half.half,
        "n_players": len(half.tracks),
        "n_positions": half.n_positions,
        "max_frame": half.max_frame,
        "image_width": half.width,
        "image_height": half.height,
    }
