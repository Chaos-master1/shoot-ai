"""Tests for kabsch algorithm (rigid alignment)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_kb = load_service_module("kb_test", "kabsch.py", subdir="utils")
kabsch_rotation = _kb.kabsch_rotation
kabsch_align = _kb.kabsch_align
kabsch_align_2d = _kb.kabsch_align_2d
apply_rigid_transform = _kb.apply_rigid_transform

import numpy as np
import pytest


class TestKabschRotation:
    def test_identity_rotation(self) -> None:
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        q = p.copy()
        R, c_src, c_tgt = kabsch_rotation(p, q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(c_src, c_tgt, atol=1e-6)

    def test_translation_doesnt_change_rotation(self) -> None:
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        q = p + np.array([10, 20, 30])
        R, c_src, c_tgt = kabsch_rotation(p, q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(c_tgt - c_src, [10, 20, 30], atol=1e-6)

    def test_90_degree_rotation_z(self) -> None:
        p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        q = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        R, _, _ = kabsch_rotation(p, q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(R, expected, atol=1e-6)

    def test_180_degree_rotation(self) -> None:
        p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        q = -p
        R, _, _ = kabsch_rotation(p, q)
        # SVD can pick a sign; verify R @ R.T == I (orthogonal)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_different_size_raises(self) -> None:
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        q = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            kabsch_rotation(p, q)

    def test_minimum_three_points(self) -> None:
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        q = p + np.array([1.0, 1.0, 0.0])
        R, _, _ = kabsch_rotation(p, q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


class TestKabschAlign:
    def test_aligned_returns_zero(self) -> None:
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        q = p.copy()
        R, t, rmsd = kabsch_align(p, q)
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-6)
        assert rmsd < 1e-6

    def test_translation(self) -> None:
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        q = p + np.array([5, 7, 9])
        R, t, rmsd = kabsch_align(p, q)
        np.testing.assert_allclose(t, [5, 7, 9], atol=1e-6)
        assert rmsd < 1e-6


class TestKabschAlign2D:
    def test_identity(self) -> None:
        p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        q = p.copy()
        angle, t, rmsd = kabsch_align_2d(p, q)
        assert abs(angle) < 1e-6
        np.testing.assert_allclose(t, [0, 0], atol=1e-6)

    def test_rotation(self) -> None:
        p = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
        R = np.array([[c, -s], [s, c]])
        q = p @ R.T
        angle, t, rmsd = kabsch_align_2d(p, q)
        assert abs(angle - math.pi / 2) < 1e-6


class TestApplyRigidTransform:
    def test_identity_transform(self) -> None:
        identity = np.eye(3)
        result = apply_rigid_transform(np.array([1.0, 2.0, 3.0]), identity, np.zeros(3))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-6)

    def test_translation_only(self) -> None:
        identity = np.eye(3)
        result = apply_rigid_transform(np.array([1.0, 2.0, 3.0]), identity, np.array([10, 20, 30]))
        np.testing.assert_allclose(result, [11.0, 22.0, 33.0], atol=1e-6)

    def test_rotation_then_translation(self) -> None:
        c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        result = apply_rigid_transform(np.array([1.0, 0.0, 0.0]), R, np.zeros(3))
        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 1.0, atol=1e-6)
