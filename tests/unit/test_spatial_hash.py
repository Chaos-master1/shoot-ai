"""Tests for spatial hash — O(1) neighbor lookups in 2D and 3D."""

import pytest

from kawkab.utils.spatial_hash import (
    SpatialHash2D,
    SpatialHash3D,
    bulk_insert_2d,
)


class TestSpatialHash2D:
    def test_empty_grid(self):
        sh = SpatialHash2D(cell_size=10.0)
        assert len(sh) == 0
        assert sh.query_cell(5, 5) == []
        assert sh.query_neighbors(5, 5, radius=5) == []

    def test_insert_and_query_cell(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("obj1", (5.0, 3.0))
        assert len(sh) == 1
        assert "obj1" in sh.query_cell(5.0, 3.0)

    def test_insert_multiple_same_cell(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("a", (1, 1))
        sh.insert("b", (2, 2))
        assert len(sh) == 2
        cell = sh.query_cell(1, 1)
        assert "a" in cell
        assert "b" in cell

    def test_insert_different_cells(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("left", (1, 1))
        sh.insert("right", (21, 1))
        assert sh.query_cell(1, 1) == ["left"]
        assert sh.query_cell(21, 1) == ["right"]

    def test_negative_cell_coords(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("neg", (-5, -5))
        assert "neg" in sh.query_cell(-5, -5)

    def test_query_neighbors_within_radius(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("a", (0, 0))
        sh.insert("b", (5, 5))
        neighbors = sh.query_neighbors(0, 0, radius=10)
        assert "a" in neighbors
        assert "b" in neighbors

    def test_query_neighbors_outside_radius(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("a", (0, 0))
        sh.insert("b", (50, 50))
        neighbors = sh.query_neighbors(0, 0, radius=5)
        assert "a" in neighbors
        assert "b" not in neighbors

    def test_query_neighbors_negative_radius_raises(self):
        sh = SpatialHash2D(cell_size=10.0)
        with pytest.raises(ValueError):
            sh.query_neighbors(0, 0, radius=-1)

    def test_invalid_cell_size_raises(self):
        with pytest.raises(ValueError):
            SpatialHash2D(cell_size=0)
        with pytest.raises(ValueError):
            SpatialHash2D(cell_size=-5)

    def test_insert_invalid_position_raises(self):
        sh = SpatialHash2D(cell_size=10.0)
        with pytest.raises(ValueError):
            sh.insert("bad", (1.0,))

    def test_clear(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("a", (1, 1))
        sh.insert("b", (2, 2))
        assert len(sh) == 2
        sh.clear()
        assert len(sh) == 0

    def test_query_empty_cell(self):
        sh = SpatialHash2D(cell_size=10.0)
        assert sh.query_cell(999, 999) == []

    def test_many_objects_same_cell(self):
        sh = SpatialHash2D(cell_size=10.0)
        for i in range(100):
            sh.insert(f"p{i}", (5, 5))
        assert len(sh) == 100
        assert len(sh.query_cell(5, 5)) == 100

    def test_radius_spans_multiple_cells(self):
        sh = SpatialHash2D(cell_size=10.0)
        sh.insert("c0", (0, 0))
        sh.insert("c1", (12, 0))
        sh.insert("c2", (-8, 0))
        n = sh.query_neighbors(0, 0, radius=15)
        assert len(n) == 3


class TestSpatialHash3D:
    def test_empty_grid(self):
        sh = SpatialHash3D(cell_size=10.0)
        assert len(sh) == 0

    def test_insert_and_query_cell(self):
        sh = SpatialHash3D(cell_size=10.0)
        sh.insert("obj", (5, 5, 5))
        assert "obj" in sh.query_cell(5, 5, 5)

    def test_insert_multiple_cells(self):
        sh = SpatialHash3D(cell_size=10.0)
        sh.insert("a", (1, 1, 1))
        sh.insert("b", (21, 1, 1))
        assert "a" in sh.query_cell(1, 1, 1)
        assert "b" in sh.query_cell(21, 1, 1)

    def test_query_neighbors_3d(self):
        sh = SpatialHash3D(cell_size=10.0)
        sh.insert("a", (0, 0, 0))
        sh.insert("b", (5, 5, 5))
        n = sh.query_neighbors(0, 0, 0, radius=10)
        assert "a" in n
        assert "b" in n

    def test_negative_radius_raises(self):
        sh = SpatialHash3D(cell_size=10.0)
        with pytest.raises(ValueError):
            sh.query_neighbors(0, 0, 0, radius=-1)

    def test_invalid_cell_size_raises(self):
        with pytest.raises(ValueError):
            SpatialHash3D(cell_size=0)

    def test_clear(self):
        sh = SpatialHash3D(cell_size=10.0)
        sh.insert("a", (1, 1, 1))
        sh.clear()
        assert len(sh) == 0

    def test_insert_invalid_position_raises(self):
        sh = SpatialHash3D(cell_size=10.0)
        with pytest.raises(ValueError):
            sh.insert("bad", (1.0, 2.0))


class TestBulkInsert:
    def test_bulk_insert_no_objs(self):
        sh = SpatialHash2D(cell_size=10.0)
        inserted = bulk_insert_2d(sh, [(1, 1), (2, 2)])
        assert len(inserted) == 2
        assert len(sh) == 2

    def test_bulk_insert_with_objs(self):
        sh = SpatialHash2D(cell_size=10.0)
        inserted = bulk_insert_2d(sh, [(1, 1), (2, 2)], objs=["a", "b"])
        assert inserted == ["a", "b"]
        assert "a" in sh.query_cell(1, 1)

    def test_bulk_insert_mismatched_lengths_raises(self):
        sh = SpatialHash2D(cell_size=10.0)
        with pytest.raises(ValueError):
            bulk_insert_2d(sh, [(1, 1), (2, 2)], objs=["a"])

    def test_bulk_insert_empty(self):
        sh = SpatialHash2D(cell_size=10.0)
        inserted = bulk_insert_2d(sh, [], [])
        assert inserted == []
        assert len(sh) == 0
