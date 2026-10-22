"""Kabsch algorithm for optimal rigid alignment between two point sets.

Given two corresponding point sets P (N x 3) and Q (N x 3), finds the
rotation R and translation t such that ||Q - (R @ P + t)||^2 is minimized.

This is the same algorithm used in zalo/MathUtilities' Kabsch.cs and is
useful for:
- Aligning broadcast tracking with pitch homography
- Sub-frame alignment of broadcast frames
- Re-identification based on body landmark matching

Reference: Kabsch, W. (1976). "A solution for the best rotation to relate
two sets of vectors". Acta Crystallographica. 32 (5): 922-923.
"""

from __future__ import annotations

import numpy as np


def kabsch_rotation(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the optimal rotation that aligns source to target.

    Both inputs must have the same shape (N, 3) and N >= 3 (non-collinear).
    Points are not centered — this function does NOT apply a translation;
    use `kabsch_align` for full rigid alignment.

    Args:
        source: (N, 3) source points
        target: (N, 3) target points (corresponding to source)

    Returns:
        rotation: (3, 3) rotation matrix
        centroid_source: (3,) centroid of source
        centroid_target: (3,) centroid of target
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError(
            f"source and target must have shape (N, 3), got {source.shape} and {target.shape}"
        )
    if source.shape[0] < 3:
        raise ValueError(f"need at least 3 points, got {source.shape[0]}")
    centroid_source = source.mean(axis=0)
    centroid_target = target.mean(axis=0)
    centered_source = source - centroid_source
    centered_target = target - centroid_target
    cov = centered_source.T @ centered_target
    u, _, vh = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vh.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vh.T @ correction @ u.T
    return rotation, centroid_source, centroid_target


def kabsch_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Find optimal rigid transform (R, t) that maps source onto target.

    Args:
        source: (N, 3) source points
        target: (N, 3) target points

    Returns:
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector
        rmsd: root-mean-square deviation of the aligned points
    """
    rotation, centroid_source, centroid_target = kabsch_rotation(source, target)
    translation = centroid_target - rotation @ centroid_source
    aligned = (rotation @ source.T).T + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, rmsd


def kabsch_align_2d(
    source_xy: np.ndarray, target_xy: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """2D version of kabsch_align for planar point sets.

    Args:
        source_xy: (N, 2) source points
        target_xy: (N, 2) target points

    Returns:
        angle: rotation angle in radians (counter-clockwise)
        translation: (2,) translation vector
        rmsd: root-mean-square deviation
    """
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape != target_xy.shape or source_xy.shape[1] != 2:
        raise ValueError(
            f"source_xy and target_xy must have shape (N, 2), got {source_xy.shape} and {target_xy.shape}"
        )
    if source_xy.shape[0] < 2:
        raise ValueError(f"need at least 2 points, got {source_xy.shape[0]}")
    src3 = np.column_stack([source_xy, np.zeros(len(source_xy))])
    tgt3 = np.column_stack([target_xy, np.zeros(len(target_xy))])
    rotation, translation, rmsd = kabsch_align(src3, tgt3)
    angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return angle, translation[:2], rmsd


def apply_rigid_transform(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply a rigid transform to a point set.

    Args:
        points: (N, 3) or (N, 2) point set
        rotation: (3, 3) or (2, 2) rotation matrix
        translation: (3,) or (2,) translation vector

    Returns:
        transformed: same shape as points
    """
    points = np.asarray(points, dtype=np.float64)
    return np.asarray((rotation @ points.T).T + translation, dtype=np.float64)
