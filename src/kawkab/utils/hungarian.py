"""Hungarian algorithm for optimal assignment (minimum cost matching).

Solves the Linear Assignment Problem: given an N x M cost matrix, find the
optimal 1-to-1 assignment between rows and columns that minimizes total cost.

This is a port of Roy Jonker's famous solution (used in zalo/MathUtilities)
and is the optimal alternative to greedy assignment used in many tracking
algorithms. Norfair uses greedy matching; using Hungarian improves ID
consistency for multi-object tracking.

Reference: Jonker, R., & Volgenant, A. (1987). "A shortest augmenting path
algorithm for dense and sparse linear assignment problems". Computing, 38(4).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def hungarian(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the optimal assignment that minimizes total cost.

    Time complexity: O(n^3) where n = max(N, M).

    Args:
        cost_matrix: (N, M) cost matrix. Can be non-square; in that case
            extra rows/columns are treated as dummies with zero cost.

    Returns:
        row_ind: (n,) row indices of the assignment
        col_ind: (n,) column indices of the assignment
        total_cost: sum of the assigned costs
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got {cost_matrix.ndim}D")
    if cost_matrix.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp), 0.0
    n, m = cost_matrix.shape
    n_max = max(n, m)
    padded = np.zeros((n_max, n_max), dtype=np.float64)
    padded[:n, :m] = cost_matrix
    u = np.zeros(n_max + 1, dtype=np.float64)
    v = np.zeros(n_max + 1, dtype=np.float64)
    p = np.zeros(n_max + 1, dtype=np.intp)
    way = np.zeros(n_max + 1, dtype=np.intp)
    for i in range(1, n_max + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_max + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_max + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_max + 1):
                if not used[j]:
                    cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_max + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    row_list: list[int] = []
    col_list: list[int] = []
    for j in range(1, m + 1):
        if p[j] != 0 and p[j] <= n:
            row_list.append(p[j] - 1)
            col_list.append(j - 1)
    if len(row_list) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp), 0.0
    row_ind = np.array(row_list, dtype=np.intp)
    col_ind = np.array(col_list, dtype=np.intp)
    total_cost = float(cost_matrix[row_ind, col_ind].sum())
    return row_ind, col_ind, total_cost


def hungarian_match(
    predictions: np.ndarray,
    detections: np.ndarray,
    cost_fn: Callable[[np.ndarray, np.ndarray], float],
    max_cost: float = np.inf,
) -> list[tuple[int, int]]:
    """Match predictions to detections using Hungarian assignment.

    Args:
        predictions: (P, D) prediction array
        detections: (N, D) detection array
        cost_fn: callable that takes (pred, det) and returns a scalar cost
        max_cost: assignments with cost above this are rejected

    Returns:
        list of (pred_idx, det_idx) tuples
    """
    predictions = np.asarray(predictions)
    detections = np.asarray(detections)
    if len(predictions) == 0 or len(detections) == 0:
        return []
    cost = np.zeros((len(predictions), len(detections)), dtype=np.float64)
    for i, pred in enumerate(predictions):
        for j, det in enumerate(detections):
            cost[i, j] = cost_fn(pred, det)
    row_ind, col_ind, _ = hungarian(cost)
    matches = []
    for r, c in zip(row_ind, col_ind, strict=False):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
    return matches
