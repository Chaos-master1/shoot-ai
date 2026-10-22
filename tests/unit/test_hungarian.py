"""Tests for Hungarian algorithm — optimal linear assignment."""

import numpy as np
import pytest

from kawkab.utils.hungarian import hungarian, hungarian_match


class TestHungarian:
    def test_square_matrix(self):
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert len(row_ind) == len(col_ind)
        assert total > 0

    def test_square_3x3(self):
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert len(row_ind) == len(col_ind)
        assert total > 0

    def test_zero_cost_matrix(self):
        cost = np.zeros((3, 3), dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert total == 0.0

    def test_non_square_more_rows(self):
        cost = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert len(row_ind) <= cost.shape[0]
        assert total > 0

    def test_non_square_more_cols(self):
        cost = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert total >= 0

    def test_empty_matrix(self):
        cost = np.empty((0, 0), dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert len(row_ind) == 0
        assert total == 0.0

    def test_1x1_matrix(self):
        cost = np.array([[42.0]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert total == 42.0

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            hungarian(np.array([1, 2, 3]))

    def test_square_2x2_minimizes(self):
        cost = np.array([[10, 1], [1, 10]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert total == 2.0

    def test_single_row(self):
        cost = np.array([[5.0, 3.0, 8.0]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert len(row_ind) == 1
        assert total >= 0

    def test_single_column(self):
        cost = np.array([[5.0], [3.0], [8.0]], dtype=float)
        row_ind, col_ind, total = hungarian(cost)
        assert len(row_ind) == 1
        assert total >= 0


class TestHungarianMatch:
    def test_basic_matching(self):
        preds = np.array([[0, 0], [10, 10]], dtype=float)
        dets = np.array([[0.5, 0.5], [9.5, 9.5]], dtype=float)
        matches = hungarian_match(preds, dets, cost_fn=lambda p, d: np.linalg.norm(p - d))
        assert len(matches) == 2

    def test_matching_with_max_cost_filter(self):
        preds = np.array([[0, 0], [100, 100]], dtype=float)
        dets = np.array([[1, 1], [200, 200]], dtype=float)
        matches = hungarian_match(
            preds, dets, cost_fn=lambda p, d: np.linalg.norm(p - d), max_cost=5
        )
        assert len(matches) == 1

    def test_empty_predictions(self):
        matches = hungarian_match(np.empty((0, 2)), np.array([[0, 0]]), cost_fn=lambda p, d: 0.0)
        assert matches == []

    def test_empty_detections(self):
        matches = hungarian_match(np.array([[0, 0]]), np.empty((0, 2)), cost_fn=lambda p, d: 0.0)
        assert matches == []

    def test_euclidean_distance_cost(self):
        preds = np.array([[0, 0]], dtype=float)
        dets = np.array([[3, 4]], dtype=float)
        matches = hungarian_match(preds, dets, cost_fn=lambda p, d: float(np.linalg.norm(p - d)))
        assert len(matches) == 1
        assert matches[0] == (0, 0)

    def test_large_cost_rejected(self):
        preds = np.array([[0, 0]], dtype=float)
        dets = np.array([[100, 100]], dtype=float)
        matches = hungarian_match(
            preds, dets, cost_fn=lambda p, d: np.linalg.norm(p - d), max_cost=5
        )
        assert matches == []
