"""Tests for the SoccerNet ground-truth loader and e2e benchmark helpers.

The loader is format-strict on purpose — a silently-truncated GT
silently corrupts every downstream MOTA/IDF1 number. These tests pin:
  - strict parsing (malformed lines raise, extra columns tolerated)
  - inverted-box normalization
  - deterministic id remap (sorted → 1000+i, cross-process stable)
  - frame_limit filtering
  - GT stride alignment used by the e2e benchmark
  - fragmentation diagnostics
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def loader():
    from kawkab.core.validation.soccernet_loader import (
        discover_soccernet_games,
        fragmentation_stats,
        load_soccernet_half,
        parse_gt_line,
        summarize_half,
    )

    class _L:
        load = staticmethod(load_soccernet_half)
        parse_line = staticmethod(parse_gt_line)
        discover = staticmethod(discover_soccernet_games)
        frag = staticmethod(fragmentation_stats)
        summarize = staticmethod(summarize_half)

    return _L()


@pytest.fixture()
def gt_file(tmp_path):
    """A small synthetic gt.txt: 3 players, 6 frames."""
    lines = []
    for f in range(1, 7):
        for tid, x in ((5, 100), (2, 400), (11, 700)):
            lines.append(f"{f} {tid} {x} 200 {x + 20} 240")
    p = tmp_path / "gameA" / "1" / "gt.txt"
    p.parent.mkdir(parents=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class TestParseGtLine:
    def test_basic_row(self, loader):
        frame, tid, x1, y1, x2, y2 = loader.parse_line("12 3 10 20 110 220", 1)
        assert (frame, tid) == (12, 3)
        assert (x1, y1, x2, y2) == (10.0, 20.0, 110.0, 220.0)

    def test_extra_columns_tolerated(self, loader):
        # Jersey number / role trailing columns are ignored.
        frame, tid, *_ = loader.parse_line("12 3 10 20 110 220 9 GK", 1)
        assert (frame, tid) == (12, 3)

    def test_too_few_columns_raises(self, loader):
        from kawkab.core.validation.soccernet_loader import SoccerNetGtFormatError

        with pytest.raises(SoccerNetGtFormatError):
            loader.parse_line("12 3 10 20 110", 4)

    def test_non_numeric_raises(self, loader):
        from kawkab.core.validation.soccernet_loader import SoccerNetGtFormatError

        with pytest.raises(SoccerNetGtFormatError):
            loader.parse_line("12 x 10 20 110 220", 4)

    def test_inverted_box_normalized(self, loader):
        _, _, x1, y1, x2, y2 = loader.parse_line("1 1 110 220 10 20", 1)
        assert x1 < x2 and y1 < y2

    def test_soccernet_csv_variant(self, loader):
        """The real tracking-2023 format: CSV, left/top/width/height."""
        frame, tid, x1, y1, x2, y2 = loader.parse_line("1,1,914,855,55,172,1,-1,-1,-1", 1)
        assert frame == 1 and tid == 1
        assert (x1, y1, x2, y2) == (914.0, 855.0, 914.0 + 55.0, 855.0 + 172.0)


class TestLoadHalfCsvVariant:
    def test_real_soccernet_layout(self, loader, tmp_path):
        """game/half_dir/gt/gt.txt layout, CSV rows."""
        p = tmp_path / "SNMOT-060" / "gt" / "gt.txt"
        p.parent.mkdir(parents=True)
        p.write_text(
            "1,1,914,855,55,172,1,-1,-1,-1\n"
            "1,2,917,575,32,122,1,-1,-1,-1\n"
            "2,1,907,855,67,172,1,-1,-1,-1\n"
            "2,2,917,575,32,122,1,-1,-1,-1\n",
            encoding="utf-8",
        )
        half = loader.load(p)
        assert half.game == "SNMOT-060"
        assert sorted(half.tracks.keys()) == [1000, 1001]
        assert half.n_positions == 4
        # id sorted numerically: 1 → 1000, 2 → 1001
        assert half.tracks[1000][0] == (1, 914.0 + 27.5, 855.0 + 86.0)


class TestLoadHalf:
    def test_tracks_shape_and_remap(self, loader, gt_file):
        half = loader.load(gt_file)
        # ids sorted numerically: 2, 5, 11 → 1000, 1001, 1002
        assert sorted(half.tracks.keys()) == [1000, 1001, 1002]
        # 6 frames × 3 players
        assert half.n_positions == 18
        # Box centers: tid 2 at x 400..420 → center 410
        assert half.tracks[1000][0] == (1, 410.0, 220.0)

    def test_frame_limit(self, loader, gt_file):
        half = loader.load(gt_file, frame_limit=3)
        assert half.n_positions == 9
        assert half.max_frame == 3

    def test_deterministic_across_calls(self, loader, gt_file):
        a = loader.load(gt_file)
        b = loader.load(gt_file)
        assert a.tracks == b.tracks

    def test_missing_file_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nope" / "gt.txt")

    def test_blank_and_comment_lines_skipped(self, loader, gt_file):
        text = gt_file.read_text(encoding="utf-8")
        gt_file.write_text("# header comment\n\n" + text, encoding="utf-8")
        half = loader.load(gt_file)
        assert half.n_positions == 18

    def test_boxes_by_frame(self, loader, gt_file):
        half = loader.load(gt_file)
        frame1 = half.boxes_by_frame[1]
        assert len(frame1) == 3
        tid, x1, y1, x2, y2 = frame1[0]
        assert tid >= 1000
        assert x2 > x1 and y2 > y1


class TestStrideAlignment:
    """The e2e benchmark's GT re-indexing must match its video sampling."""

    @staticmethod
    def _align(positions, stride, frames_processed):
        out = []
        rank = 0
        for f, x, y in sorted(positions, key=lambda p: p[0]):
            if f % stride == 0:
                rank += 1
                if rank <= frames_processed:
                    out.append((rank, x, y))
        return out

    def test_stride1_identity(self, loader, gt_file):
        half = loader.load(gt_file)
        for _tid, positions in half.tracks.items():
            assert self._align(positions, 1, 6) == positions

    def test_stride2_halves_frames(self):
        positions = [
            (1, 0.0, 0.0),
            (2, 1.0, 0.0),
            (3, 2.0, 0.0),
            (4, 3.0, 0.0),
            (5, 4.0, 0.0),
            (6, 5.0, 0.0),
        ]
        aligned = self._align(positions, 2, 3)
        # GT frames 2,4,6 → processed ranks 1,2,3
        assert aligned == [(1, 1.0, 0.0), (2, 3.0, 0.0), (3, 5.0, 0.0)]

    def test_processed_shorter_than_gt(self):
        positions = [(f, float(f), 0.0) for f in range(1, 11)]
        aligned = self._align(positions, 2, 3)
        assert len(aligned) == 3
        assert aligned[-1][0] == 3

    def test_stride_alignment_keeps_absolute_time(self):
        """REGRESSION (2026-09-16, e2e benchmark): GT alignment must preserve
        WHEN a position occurred, not re-rank positions within each track.

        The old loop stamped every track's first position as processed-frame
        1, so a track entering at GT frame 57 was 'teleported' 56 frames
        earlier — past ~frame 50 every pred↔gt comparison happened at the
        wrong time and FP/FN exploded linearly with clip length (published
        MOTA −0.06 while a per-frame recount of the same data gave ~0.80).
        Correct mapping: GT frame f (f % stride == 0) → processed index
        f // stride.
        """
        # A track that only EXISTS from GT frame 57 onward.
        positions = [(f, 10.0, 20.0) for f in range(57, 61)]
        stride = 1
        frames_processed = 60
        aligned = [
            (f // stride, x, y)
            for f, x, y in sorted(positions, key=lambda p: p[0])
            if f % stride == 0 and f // stride <= frames_processed
        ]
        # First position must land on processed frame 57, NOT frame 1.
        assert aligned[0][0] == 57
        assert [a[0] for a in aligned] == [57, 58, 59, 60]

    def test_stride_alignment_with_stride(self):
        positions = [(f, float(f), 0.0) for f in range(56, 65)]  # 56..64
        stride = 2
        frames_processed = 32
        aligned = [
            (f // stride, x, y)
            for f, x, y in sorted(positions, key=lambda p: p[0])
            if f % stride == 0 and f // stride <= frames_processed
        ]
        # GT frames 56,58,60,62,64 → processed 28,29,30,31,32
        assert [a[0] for a in aligned] == [28, 29, 30, 31, 32]


class TestFragmentation:
    def test_empty(self, loader):
        assert loader.frag({})["n_pred_tracks"] == 0.0

    def test_counts_and_lengths(self, loader):
        pred = {
            1: [(i, 0.0, 0.0) for i in range(1, 11)],  # len 10
            2: [(i, 0.0, 0.0) for i in range(1, 6)],  # len 5
            3: [(i, 0.0, 0.0) for i in range(1, 9)],  # len 8
        }
        stats = loader.frag(pred)
        assert stats["n_pred_tracks"] == 3.0
        assert stats["mean_track_length"] == pytest.approx(23.0 / 3.0)
        assert stats["median_track_length"] == 8.0

    def test_fragmentation_signal(self, loader):
        """22 identities → 91 tracks means mean length collapses; the
        stat must be able to see the difference."""
        one_track_each = {i: [(f, 0.0, 0.0) for f in range(1, 101)] for i in range(22)}
        fragmented = {
            f"{i}_{k}": [(f, 0.0, 0.0) for f in range(1, 6)]
            for i in range(22)
            for k in range(4)  # 88 tracks of length 5 ≈ the README's 91
        }
        clean = loader.frag(one_track_each)
        broken = loader.frag(fragmented)
        assert broken["n_pred_tracks"] / 22 > 3 * (clean["n_pred_tracks"] / 22)
        assert clean["mean_track_length"] > 4 * broken["mean_track_length"]


class TestDiscover:
    def test_finds_both_layouts(self, loader, tmp_path):
        a = tmp_path / "g1" / "1" / "gt.txt"
        a.parent.mkdir(parents=True)
        a.write_text("", encoding="utf-8")
        b = tmp_path / "g2" / "2_gt.txt"
        b.parent.mkdir(parents=True)
        b.write_text("", encoding="utf-8")
        found = loader.discover(tmp_path)
        assert a in found and b in found

    def test_missing_root_empty(self, loader, tmp_path):
        assert loader.discover(tmp_path / "nope") == []


class TestSummarize:
    def test_summary_fields(self, loader, gt_file):
        half = loader.load(gt_file)
        s = loader.summarize(half)
        assert s["game"] == "1"
        assert s["half"] == 1
        assert s["n_players"] == 3
        assert s["n_positions"] == 18
