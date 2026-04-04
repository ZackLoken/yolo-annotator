"""Tests for AnnotationEngine — headless annotation logic."""

import os
import tempfile

import pytest

from yololabeler.state import AppState
from yololabeler.annotation.engine import AnnotationEngine


@pytest.fixture
def engine():
    state = AppState()
    state.img_width = 100
    state.img_height = 100
    return AnnotationEngine(state)


# ── Spatial index ─────────────────────────────────────────────────────────


class TestSpatialIndex:
    def test_ensure_poly_bboxes_empty(self, engine):
        engine.ensure_poly_bboxes()
        assert engine.state._poly_bboxes == []
        assert not engine.state._poly_bboxes_dirty

    def test_ensure_poly_bboxes_single(self, engine):
        engine.state.polygons = [([(10, 20), (30, 40), (50, 10)], 0)]
        engine.invalidate_poly_bboxes()
        engine.ensure_poly_bboxes()
        assert engine.state._poly_bboxes == [(10, 10, 50, 40)]

    def test_ensure_poly_bboxes_caches(self, engine):
        engine.state.polygons = [([(0, 0), (10, 10), (20, 0)], 0)]
        engine.invalidate_poly_bboxes()
        engine.ensure_poly_bboxes()
        assert not engine.state._poly_bboxes_dirty
        # Calling again should be a no-op (no rebuild)
        engine.state.polygons = []  # change data
        engine.ensure_poly_bboxes()  # should NOT rebuild
        assert len(engine.state._poly_bboxes) == 1  # still cached

    def test_invalidate_marks_dirty(self, engine):
        engine.state._poly_bboxes_dirty = False
        engine.invalidate_poly_bboxes()
        assert engine.state._poly_bboxes_dirty


# ── Undo / Redo ───────────────────────────────────────────────────────────


class TestUndoRedo:
    def test_push_undo(self, engine):
        engine.state.boxes = [(0, 10, 20, 30, 40)]
        engine.state.polygons = []
        engine.push_undo()
        assert len(engine.state._undo_stack) == 1

    def test_undo_restores_state(self, engine):
        engine.state.boxes = [(0, 1, 2, 3, 4)]
        engine.push_undo()
        engine.state.boxes = [(1, 5, 6, 7, 8)]
        assert engine.undo_snapshot()
        assert engine.state.boxes == [(0, 1, 2, 3, 4)]

    def test_undo_returns_false_when_empty(self, engine):
        assert not engine.undo_snapshot()

    def test_redo_restores_state(self, engine):
        engine.state.boxes = [(0, 1, 2, 3, 4)]
        engine.push_undo()
        engine.state.boxes = [(1, 5, 6, 7, 8)]
        engine.undo_snapshot()
        assert engine.redo_snapshot()
        assert engine.state.boxes == [(1, 5, 6, 7, 8)]

    def test_redo_returns_false_when_empty(self, engine):
        assert not engine.redo_snapshot()

    def test_undo_redo_round_trip(self, engine):
        original_boxes = [(0, 10, 20, 30, 40)]
        engine.state.boxes = list(original_boxes)
        engine.push_undo()
        engine.state.boxes = [(1, 50, 60, 70, 80)]
        engine.undo_snapshot()
        engine.redo_snapshot()
        assert engine.state.boxes == [(1, 50, 60, 70, 80)]

    def test_undo_stack_cap(self, engine):
        for i in range(35):
            engine.state.boxes = [(i, 0, 0, 0, 0)]
            engine.push_undo()
        assert len(engine.state._undo_stack) == 30

    def test_push_undo_clears_redo(self, engine):
        engine.state.boxes = [(0, 1, 2, 3, 4)]
        engine.push_undo()
        engine.state.boxes = [(1, 5, 6, 7, 8)]
        engine.undo_snapshot()
        # Redo stack has 1 entry
        assert len(engine.state._redo_stack) == 1
        engine.push_undo()
        # Redo stack cleared after new push
        assert len(engine.state._redo_stack) == 0

    def test_undo_clears_drag_state(self, engine):
        engine.state._dragging_vertex = (0, 1)
        engine.state._drag_orig_pos = (10.0, 20.0)
        engine.state.boxes = [(0, 1, 2, 3, 4)]
        engine.push_undo()
        engine.state.boxes = []
        engine.undo_snapshot()
        assert engine.state._dragging_vertex is None
        assert engine.state._drag_orig_pos is None


# ── Clear drag state ──────────────────────────────────────────────────────


class TestClearDragState:
    def test_resets_all(self, engine):
        engine.state._dragging_vertex = (0, 2)
        engine.state._drag_orig_pos = (5.0, 5.0)
        engine.state._hovered_polygon_idx = 3
        engine.clear_drag_state()
        assert engine.state._dragging_vertex is None
        assert engine.state._drag_orig_pos is None
        assert engine.state._hovered_polygon_idx is None


# ── Close current polygon ────────────────────────────────────────────────


class TestCloseCurrentPolygon:
    def test_closes_valid_polygon(self, engine):
        engine.state.current_polygon = [(10, 10), (50, 10), (30, 50)]
        engine.state.active_class = 2
        assert engine.close_current_polygon()
        assert len(engine.state.polygons) == 1
        points, cls = engine.state.polygons[0]
        assert cls == 2
        assert len(points) == 3
        assert engine.state.current_polygon == []

    def test_rejects_too_few_vertices(self, engine):
        engine.state.current_polygon = [(10, 10), (50, 10)]
        assert not engine.close_current_polygon()
        assert engine.state.polygons == []
        assert engine.state.current_polygon == []

    def test_clamps_to_image_bounds(self, engine):
        engine.state.current_polygon = [(-5, -10), (150, 200), (50, 50)]
        engine.close_current_polygon()
        points, _ = engine.state.polygons[0]
        assert points[0] == (0, 0)
        assert points[1] == (100, 100)
        assert points[2] == (50, 50)

    def test_pushes_undo(self, engine):
        engine.state.current_polygon = [(10, 10), (50, 10), (30, 50)]
        engine.close_current_polygon()
        assert len(engine.state._undo_stack) == 1

    def test_invalidates_spatial_index(self, engine):
        engine.state._poly_bboxes_dirty = False
        engine.state.current_polygon = [(10, 10), (50, 10), (30, 50)]
        engine.close_current_polygon()
        assert engine.state._poly_bboxes_dirty


# ── Save ──────────────────────────────────────────────────────────────────


class TestSave:
    def test_save_creates_files(self, engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = engine.state
            s.images = ["test.png"]
            s.index = 0
            s.labels_dir = os.path.join(tmpdir, "labels")
            s.detect_dir = os.path.join(s.labels_dir, "detect")
            s.segment_dir = os.path.join(s.labels_dir, "segment")
            s.img_width = 640
            s.img_height = 480
            s.boxes = [(0, 100, 100, 200, 200)]
            s.polygons = [([(10, 10), (50, 10), (30, 50)], 0)]
            assert engine.save()
            assert os.path.exists(
                os.path.join(s.detect_dir, "test.txt"))
            assert os.path.exists(
                os.path.join(s.segment_dir, "test.txt"))

    def test_save_returns_false_no_images(self, engine):
        assert not engine.save()

    def test_save_returns_false_zero_dims(self, engine):
        engine.state.images = ["test.png"]
        engine.state.index = 0
        engine.state.labels_dir = "/tmp"
        engine.state.detect_dir = "/tmp"
        engine.state.segment_dir = "/tmp"
        engine.state.img_width = 0
        engine.state.img_height = 0
        assert not engine.save()
