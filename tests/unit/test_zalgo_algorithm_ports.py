"""Tests for the algorithm ports (kabsch, hungarian, spatial_hash) from zalo/MathUtilities."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_loguru_stub, load_service_module

install_loguru_stub()
_kab = load_service_module("kabsch_test", "kabsch.py", subdir="utils")
_hun = load_service_module("hun_test", "hungarian.py", subdir="utils")
_sha = load_service_module("sha_test", "spatial_hash.py", subdir="utils")

kabsch_align = _kab.kabsch_align
kabsch_align_2d = _kab.kabsch_align_2d
kabsch_rotation = _kab.kabsch_rotation
apply_rigid_transform = _kab.apply_rigid_transform
hungarian = _hun.hungarian
hungarian_match = _hun.hungarian_match
SpatialHash2D = _sha.SpatialHash2D
SpatialHash3D = _sha.SpatialHash3D
bulk_insert_2d = _sha.bulk_insert_2d

import numpy as np
import pytest


class TestKabschRotation:
    def test_self_alignment_zero_error(self) -> None:
        src = np.random.randn(20, 3)
        R, c_src, c_tgt = kabsch_rotation(src, src)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(c_src, c_tgt)

    def test_recovers_rotation(self) -> None:
        np.random.seed(42)
        src = np.random.randn(50, 3)
        angle = 0.6
        R_true = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        tgt = (R_true @ src.T).T
        R, _, _ = kabsch_rotation(src, tgt)
        np.testing.assert_allclose(R, R_true, atol=1e-6)

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            kabsch_rotation(np.zeros((5, 3)), np.zeros((6, 3)))

    def test_minimum_points(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            kabsch_rotation(np.zeros((2, 3)), np.zeros((2, 3)))


class TestKabschAlign:
    def test_identity_no_translation(self) -> None:
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        R, t, rmsd = kabsch_align(src, src)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-9)
        assert rmsd < 1e-9

    def test_translation_recovery(self) -> None:
        src = np.random.randn(10, 3)
        t_true = np.array([3.0, -2.0, 5.0])
        tgt = src + t_true
        _, t, rmsd = kabsch_align(src, tgt)
        np.testing.assert_allclose(t, t_true, atol=1e-9)
        assert rmsd < 1e-9

    def test_full_transform_recovery(self) -> None:
        np.random.seed(7)
        src = np.random.randn(20, 3)
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t_true = np.array([10.0, 20.0, 30.0])
        tgt = (R_true @ src.T).T + t_true
        R, t, rmsd = kabsch_align(src, tgt)
        np.testing.assert_allclose(R, R_true, atol=1e-6)
        np.testing.assert_allclose(t, t_true, atol=1e-6)
        assert rmsd < 1e-9


class TestKabschAlign2D:
    def test_2d_identity(self) -> None:
        src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        angle, t, rmsd = kabsch_align_2d(src, src)
        assert abs(angle) < 1e-6
        np.testing.assert_allclose(t, [0, 0], atol=1e-6)
        assert rmsd < 1e-6

    def test_2d_rotation_recovery(self) -> None:
        src = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        angle = np.pi / 4
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        tgt = (R @ src.T).T
        recovered, _, _ = kabsch_align_2d(src, tgt)
        assert abs(recovered - angle) < 1e-6


class TestApplyRigidTransform:
    def test_3d(self) -> None:
        pts = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        R = np.eye(3)
        t = np.array([1, 2, 3], dtype=float)
        result = apply_rigid_transform(pts, R, t)
        np.testing.assert_allclose(result, pts + t, atol=1e-9)


class TestHungarian:
    def test_simple_3x3(self) -> None:
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
        rows, cols, total = hungarian(cost)
        # Optimal: (0,1) + (1,0) + (2,2) = 1 + 2 + 2 = 5
        assert total == 5.0
        assert set(rows.tolist()) == {0, 1, 2}
        assert set(cols.tolist()) == {0, 1, 2}

    def test_identity_2x2(self) -> None:
        cost = np.array([[1, 5], [3, 1]], dtype=float)
        rows, cols, total = hungarian(cost)
        # (0,0)=1 + (1,1)=1 = 2
        assert total == 2.0

    def test_non_square(self) -> None:
        cost = np.array([[10, 20], [30, 5]], dtype=float)
        rows, cols, total = hungarian(cost)
        # (0,0)=10 + (1,1)=5 = 15
        assert total == 15.0

    def test_empty(self) -> None:
        rows, cols, total = hungarian(np.zeros((0, 0)))
        assert total == 0.0

    def test_match_helper(self) -> None:
        preds = np.array([[0, 0], [10, 10], [20, 20]])
        dets = np.array([[0.5, 0.5], [9, 11], [25, 25]])
        matches = hungarian_match(preds, dets, lambda p, d: np.linalg.norm(p - d), max_cost=2.0)
        # pred 0 -> det 0 (dist 0.7), pred 1 -> det 1 (dist 1.4), pred 2 unmatched
        assert (0, 0) in matches
        assert (1, 1) in matches
        assert (2, 2) not in matches

    def test_high_cost_rejected(self) -> None:
        preds = np.array([[0, 0]])
        dets = np.array([[100, 100]])
        matches = hungarian_match(preds, dets, lambda p, d: np.linalg.norm(p - d), max_cost=1.0)
        assert matches == []


class TestSpatialHash2D:
    def test_empty(self) -> None:
        sh = SpatialHash2D()
        assert len(sh) == 0

    def test_insert_and_query(self) -> None:
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("a", (1, 1))
        sh.insert("b", (50, 50))  # different cell
        sh.insert("c", (200, 200))  # different cell
        # Both a and b in same cell (1,1) and (50,50) are far apart, but (2,2) and (1,1) might share a cell
        cell_a = sh.query_cell(1, 1)
        assert "a" in cell_a
        # Far away items not in nearby query
        neighbors = sh.query_neighbors(1, 1, radius=5)
        assert "a" in neighbors
        assert "b" not in neighbors
        assert "c" not in neighbors

    def test_clear(self) -> None:
        sh = SpatialHash2D()
        sh.insert("a", (1, 1))
        sh.insert("b", (2, 2))
        assert len(sh) == 2
        sh.clear()
        assert len(sh) == 0

    def test_cell_size_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SpatialHash2D(cell_size=0)

    def test_negative_radius(self) -> None:
        sh = SpatialHash2D()
        with pytest.raises(ValueError, match="non-negative"):
            sh.query_neighbors(0, 0, radius=-1)

    def test_insert_2d_only(self) -> None:
        sh = SpatialHash2D()
        sh.insert("a", (1, 2, 3))  # 3D pos, only first 2 used
        assert sh.query_cell(1, 2) == ["a"]

    def test_bulk_insert_2d(self) -> None:
        sh = SpatialHash2D(cell_size=10.0)
        positions = [(0, 0), (5, 5), (15, 15)]
        bulk_insert_2d(sh, positions)
        assert len(sh) == 3


class TestSpatialHash3D:
    def test_empty(self) -> None:
        sh = SpatialHash3D()
        assert len(sh) == 0

    def test_insert_and_query(self) -> None:
        sh = SpatialHash3D(cell_size=10.0)
        sh.insert("a", (1, 1, 1))
        sh.insert("b", (50, 50, 50))  # far away
        sh.insert("c", (200, 200, 200))  # very far
        neighbors = sh.query_neighbors(1, 1, 1, radius=5)
        assert "a" in neighbors
        assert "b" not in neighbors
        assert "c" not in neighbors

    def test_cell_size_validation(self) -> None:
        with pytest.raises(ValueError):
            SpatialHash3D(cell_size=0)

    def test_clear(self) -> None:
        sh = SpatialHash3D()
        sh.insert("a", (0, 0, 0))
        sh.clear()
        assert len(sh) == 0

    def test_2d_position_rejected(self) -> None:
        sh = SpatialHash3D()
        with pytest.raises(ValueError):
            sh.insert("a", (1, 2))  # only 2 coords
