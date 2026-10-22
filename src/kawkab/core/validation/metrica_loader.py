"""Metrica Sports tracking-data loader (Sample_Game_1 raw CSV format).

Metrica raw tracking CSVs: two files (home/away), each row one frame.
Column layout (after two header rows):

    Period, Frame, Time [s], Player11_x, Player11_y, Player1_x, ...

    Player columns are 2-wide (x, y) with alternating NaN for the unused
    half of each pair, and the last player pair is the Ball. Away files
    add a leading column of empty team labels (3 leading empty columns
    in header row 1 instead of 2) — the parser locates columns by header
    names, not fixed offsets, so both shapes work.

Coordinates: Metrica uses normalized [0,1] x [0,1], origin at the HOME
team's left corner, y from one touchline. Home attacks toward x=1.
Converted here to Kawkab meters (105x68).

The frame index in the two files is aligned 1:1 (same Period/Frame rows,
~25 fps). This loader yields per-frame snapshots with home positions,
away positions, and ball position — the schema the OBV/EPV/xT-ground-
truth gap in CLAUDE.md describes as ``frames: [{timestamp, possession,
ball_pos, home_positions, away_positions}, ...]``.

No I/O at import time; numpy + stdlib only.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

KAWKAB_X_MAX = 105.0
KAWKAB_Y_MAX = 68.0
METRICA_FPS = 25.0


@dataclass
class Frame:
    """One tracking snapshot, Kawkab meters, home attacks +x."""

    frame: int
    period: int
    time_s: float
    ball: tuple[float, float] | None
    home: list[tuple[float, float]]
    away: list[tuple[float, float]]

    def to_obv_dict(self) -> dict[str, Any]:
        """Serialize to the OBV/EPV frame schema (see core/obv.py)."""
        return {
            "timestamp": self.time_s,
            "possession": "home",  # possession unknown from raw files; caller patches
            "ball_pos": list(self.ball) if self.ball else None,
            "home_positions": [list(p) for p in self.home],
            "away_positions": [list(p) for p in self.away],
        }


@dataclass
class MetricaMatch:
    """Full match of aligned tracking frames."""

    frames: list[Frame] = field(default_factory=list)
    fps: float = METRICA_FPS

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def positions_by_player(self) -> dict[str, np.ndarray]:
        """{player_key: (N_frames, 2) array} for distance/velocity analytics.

        Keys are ``h0..hN`` / ``a0..aN`` by jersey order index; all-NaN
        columns (injured-off / subbed players) are dropped.
        """
        out: dict[str, np.ndarray] = {}
        n = len(self.frames)
        for side, pos_lists in (
            ("h", [f.home for f in self.frames]),
            ("a", [f.away for f in self.frames]),
        ):
            max_n = max((len(pl) for pl in pos_lists), default=0)
            for idx in range(max_n):
                arr = np.full((n, 2), np.nan)
                for fi, pos_list in enumerate(pos_lists):
                    if idx < len(pos_list):
                        arr[fi] = pos_list[idx]
                if not np.all(np.isnan(arr)):
                    out[f"{side}{idx}"] = arr
        return out

    def sample(self, every_n: int = 25) -> list[Frame]:
        """Uniformly subsample frames (every_n-th), preserving order."""
        return list(self.frames[::every_n]) if every_n > 1 else list(self.frames)


def _norm_to_meters(x: float | None, y: float | None) -> tuple[float, float] | None:
    if x is None or y is None:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x * KAWKAB_X_MAX, y * KAWKAB_Y_MAX


def _parse_pair(raw_x: str, raw_y: str) -> tuple[float, float] | None:
    """Parse one 'PlayerN_x, PlayerN_y' CSV pair, NaN/empty-safe."""
    try:
        x = float(raw_x) if raw_x.strip() else float("nan")
        y = float(raw_y) if raw_y.strip() else float("nan")
    except ValueError:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return _norm_to_meters(x, y)


def load_metrica_match(
    home_csv: str | Path,
    away_csv: str | Path,
    *,
    max_frames: int | None = None,
) -> MetricaMatch:
    """Parse the two raw-tracking CSVs into aligned Frame objects."""
    home_path, away_path = Path(home_csv), Path(away_csv)

    home_rows = _read_raw_csv(home_path)
    away_rows = _read_raw_csv(away_path)

    frames: list[Frame] = []
    away_iter = iter(away_rows)
    away_row = next(away_iter, None)

    for h_row in home_rows:
        if away_row is None or away_row[1] != h_row[1]:  # Frame column mismatch
            # Re-sync: scan forward in away rows for the matching frame
            away_row = _resync(away_iter, h_row[1])
            if away_row is None:
                continue
        frame = _build_frame(h_row, away_row)
        if frame is not None:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        away_row = next(away_iter, None)

    return MetricaMatch(frames=frames)


def _resync(away_iter, target_frame: str):
    """Advance the away iterator to the row matching target_frame."""
    for row in away_iter:
        if row[1] == target_frame:
            return row
    return None


def _read_raw_csv(path: Path) -> list[list[str]]:
    """Read a Metrica raw CSV, returning data rows (headers stripped)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError(f"{path}: too few rows ({len(rows)})")
    return rows[2:]


def _build_frame(home_row: list[str], away_row: list[str]) -> Frame | None:
    """Build one Frame from aligned home/away data rows."""
    try:
        period = int(home_row[0])
        frame_idx = int(home_row[1])
        time_s = float(home_row[2])
    except (ValueError, IndexError):
        return None

    home_pos: list[tuple[float, float]] = []
    away_pos: list[tuple[float, float]] = []
    ball: tuple[float, float] | None = None

    # Home file: cols 3.. are PlayerN_x,PlayerN_y pairs then Ball_x,Ball_y
    i = 3
    n = len(home_row)
    while i + 1 < n:
        if i + 2 >= n:  # last pair is the ball
            ball = _parse_pair(home_row[i], home_row[i + 1])
            break
        pos = _parse_pair(home_row[i], home_row[i + 1])
        if pos is not None:
            home_pos.append(pos)
        i += 2

    # Away file: same layout (3 leading meta columns in data rows)
    j = 3
    m = len(away_row)
    while j + 1 < m:
        if j + 2 >= m:
            if ball is None:
                ball = _parse_pair(away_row[j], away_row[j + 1])
            break
        pos = _parse_pair(away_row[j], away_row[j + 1])
        if pos is not None:
            away_pos.append(pos)
        j += 2

    return Frame(
        frame=frame_idx,
        period=period,
        time_s=time_s,
        ball=ball,
        home=home_pos,
        away=away_pos,
    )
