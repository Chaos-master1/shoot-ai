"""Tests for Kabsch algorithm — optimal rigid alignment of point sets."""

import numpy as np
import pytest

from kawkab.utils.kabsch import (
    apply_rigid_transform,
    kabsch_align,
    kabsch_align_2d,
    kabsch_rotation,
)


class TestKabschRotation:
    def test_identity_transform(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        rot, cs, ct = kabsch_rotation(src, src)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
        assert np.allclose(np.linalg.det(rot), 1.0, atol=1e-10)

    def test_90_degree_rotation_xy(self):
        src = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=float)
        tgt = np.array([[0, 1, 0], [0, 0, 0], [-1, 0, 0]], dtype=float)
        rot, cs, ct = kabsch_rotation(src, tgt)
        aligned = (rot @ src.T).T + (ct - rot @ cs)
        assert np.allclose(aligned, tgt, atol=1e-10)

    def test_minimum_three_points(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tgt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        rot, cs, ct = kabsch_rotation(src, tgt)
        assert rot.shape == (3, 3)

    def test_fewer_than_three_points_raises(self):
        src = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        tgt = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        with pytest.raises(ValueError):
            kabsch_rotation(src, tgt)

    def test_wrong_shape_raises(self):
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        tgt = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        with pytest.raises(ValueError):
            kabsch_rotation(src, tgt)

    def test_mismatched_shapes_raises(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tgt = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        with pytest.raises(ValueError):
            kabsch_rotation(src, tgt)

    def test_centroid_output(self):
        src = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        rot, cs, ct = kabsch_rotation(src, src)
        assert np.allclose(cs, src.mean(axis=0))
        assert np.allclose(ct, src.mean(axis=0))


class TestKabschAlign:
    def test_identity_alignment(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        rot, trans, rmsd = kabsch_align(src, src)
        assert rmsd == pytest.approx(0.0, abs=1e-10)
        assert np.allclose(rot, np.eye(3), atol=1e-10)

    def test_translation_only(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tgt = src + np.array([10, 20, 30])
        rot, trans, rmsd = kabsch_align(src, tgt)
        assert rmsd < 1e-10
        assert np.allclose(rot, np.eye(3), atol=1e-10)
        assert np.allclose(trans, [10, 20, 30])

    def test_rotation_and_translation(self):
        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        src = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        tgt = (rot_z @ src.T).T + np.array([5, 5, 0])
        rot, trans, rmsd = kabsch_align(src, tgt)
        assert rmsd < 1e-10

    def test_rmsd_non_zero(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tgt = np.array([[0.1, 0, 0], [1, 0.1, 0], [0, 1, 0.1]], dtype=float)
        rot, trans, rmsd = kabsch_align(src, tgt)
        assert rmsd > 0

    def test_rmsd_property(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tgt = np.array([[0.5, 0, 0], [1.5, 0, 0], [0.5, 1, 0]], dtype=float)
        rot, trans, rmsd = kabsch_align(src, tgt)
        aligned = (rot @ src.T).T + trans
        expected_rmsd = float(np.sqrt(np.mean(np.sum((aligned - tgt) ** 2, axis=1))))
        assert np.isclose(rmsd, expected_rmsd)


class TestKabschAlign2D:
    def test_identity_2d(self):
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        angle, trans, rmsd = kabsch_align_2d(src, src)
        assert rmsd == pytest.approx(0.0, abs=1e-10)
        assert np.isclose(angle, 0.0, atol=1e-10)

    def test_translation_2d(self):
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        tgt = src + np.array([10, 20])
        angle, trans, rmsd = kabsch_align_2d(src, tgt)
        assert rmsd < 1e-10
        assert np.allclose(trans, [10, 20])

    def test_rotation_90_2d(self):
        src = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
        tgt = np.array([[0, 1], [0, 0], [-1, 0]], dtype=float)
        angle, trans, rmsd = kabsch_align_2d(src, tgt)
        assert rmsd < 1e-10
        assert np.isclose(angle, np.pi / 2, atol=1e-5) or np.isclose(
            angle, -3 * np.pi / 2, atol=1e-5
        )

    def test_fewer_than_two_points_raises(self):
        src = np.array([[0, 0]], dtype=float)
        tgt = np.array([[0, 0]], dtype=float)
        with pytest.raises(ValueError):
            kabsch_align_2d(src, tgt)

    def test_wrong_shape_raises(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        tgt = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError):
            kabsch_align_2d(src, tgt)


class TestApplyRigidTransform:
    def test_identity_transform(self):
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = apply_rigid_transform(points, np.eye(3), np.zeros(3))
        assert np.allclose(result, points)

    def test_translation(self):
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        result = apply_rigid_transform(points, np.eye(3), np.array([10, 20, 30]))
        assert np.allclose(result, points + [10, 20, 30])

    def test_rotation_only(self):
        points = np.array([[1, 0, 0]], dtype=float)
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        result = apply_rigid_transform(points, rot, np.zeros(3))
        assert np.allclose(result, [[0, 1, 0]])

    def test_rotation_and_translation(self):
        points = np.array([[1, 0, 0]], dtype=float)
        rot = np.eye(3)
        result = apply_rigid_transform(points, rot, np.array([5, 5, 5]))
        assert np.allclose(result, [[6, 5, 5]])

    def test_2d_points_with_3d_transform(self):
        points = np.array([[1, 0], [0, 1]], dtype=float)
        rot = np.array([[0, -1], [1, 0]], dtype=float)
        result = apply_rigid_transform(points, rot, np.array([0, 0]))
        assert np.allclose(result, [[0, 1], [-1, 0]])
