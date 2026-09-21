"""Tests for AnnotationEngine, headless annotation logic over a Document."""

import os

import pytest

from yololabeler.annotation.document import Document, new_annotation
from yololabeler.annotation.engine import AnnotationEngine
from yololabeler.state import AppState

TRI = ((10, 10), (30, 40), (50, 10))


@pytest.fixture
def engine(tmp_path):
    state = AppState()
    state.image_folder = str(tmp_path)
    state.images = ["a.jpg"]
    state.index = 0
    state.img_width = 100
    state.img_height = 100
    state.labels_dir = str(tmp_path / "labels")
    state.detect_dir = str(tmp_path / "labels" / "detect")
    state.segment_dir = str(tmp_path / "labels" / "segment")
    state.state_dir = str(tmp_path / "state")
    state.document = Document("a.jpg", 100, 100)
    state._current_user = "zack"
    return AnnotationEngine(state)


# ── Spatial index ───────────────────────────────────────────────────────────


class TestSpatialIndex:
    def test_bboxes_keyed_by_id(self, engine):
        a = new_annotation("polygon", TRI, 0, "z")
        engine.state.document.add(a)
        engine.invalidate_poly_bboxes()
        engine.ensure_poly_bboxes()
        assert engine.state._poly_bboxes == {a.id: (10, 10, 50, 40)}

    def test_cached_until_invalidated(self, engine):
        engine.ensure_poly_bboxes()
        assert not engine.state._poly_bboxes_dirty


# ── Creation ────────────────────────────────────────────────────────────────


class TestCreation:
    def test_add_box_records_author(self, engine):
        a = engine.add_box(5, 5, 20, 20)
        assert a in engine.state.document.annotations
        assert a.kind == "box" and a.author == "zack" and a.source == "drawn"
        assert a.class_id == engine.state.active_class

    def test_close_polygon_clamps_and_clears(self, engine):
        engine.state.current_polygon = [(-5, 10), (30, 40), (150, 10)]
        a = engine.close_current_polygon()
        assert a.points == ((0.0, 10.0), (30.0, 40.0), (100.0, 10.0))
        assert engine.state.current_polygon == []

    def test_close_polygon_needs_three(self, engine):
        engine.state.current_polygon = [(0, 0), (1, 1)]
        assert engine.close_current_polygon() is None
        assert engine.state.document.annotations == []

    def test_close_polygon_discards_a_flat_one(self, engine):
        engine.state.current_polygon = [(0, 0), (10, 10), (20, 20), (30, 30)]
        assert engine.close_current_polygon() is None
        assert engine.state.document.annotations == []
        assert (
            engine.state.current_polygon == []
            and engine.state._undo_stack == []
        )


# ── Undo / redo ─────────────────────────────────────────────────────────────


class TestUndoRedo:
    def test_undo_restores_annotations_and_verdicts(self, engine):
        s = engine.state
        s.verdicts["h:1"] = {"action": "rejected"}
        engine.push_undo()
        a = engine.add_box(5, 5, 20, 20)
        s.verdicts["h:2"] = {"action": "accepted"}
        assert engine.undo_snapshot()
        assert s.document.annotations == [] and s.verdicts == {
            "h:1": {"action": "rejected"}
        }
        assert engine.redo_snapshot()
        assert s.document.annotations == [a] and "h:2" in s.verdicts

    def test_undo_keeps_verdict_dict_identity(self, engine):
        s = engine.state
        live = s.verdicts
        engine.push_undo()
        s.verdicts["k"] = {}
        engine.undo_snapshot()
        assert s.verdicts is live and live == {}

    def test_empty_stacks(self, engine):
        assert not engine.undo_snapshot() and not engine.redo_snapshot()

    def test_undo_depth(self, engine):
        for _ in range(35):
            engine.push_undo()
        assert len(engine.state._undo_stack) == 30


# ── Delete and edit ─────────────────────────────────────────────────────────


class TestEdit:
    def test_delete_and_set_points(self, engine):
        a = engine.add_box(5, 5, 20, 20)
        engine.set_points(a.id, ((6, 6), (21, 21)))
        assert engine.state.document.get(a.id).points == (
            (6.0, 6.0),
            (21.0, 21.0),
        )
        assert engine.delete_annotation(a.id).id == a.id
        assert engine.state.document.annotations == []


# ── Save ────────────────────────────────────────────────────────────────────


class TestSave:
    def test_save_writes_labels_and_sidecar(self, engine):
        engine.add_box(0, 0, 50, 50)
        assert engine.save() is None
        detect, segment, sidecar = engine.label_paths()
        assert (
            os.path.exists(detect)
            and os.path.exists(sidecar)
            and not os.path.exists(segment)
        )

    def test_save_reports_failure(self, engine):
        engine.add_box(0, 0, 50, 50)
        os.makedirs(engine.state.detect_dir, exist_ok=True)
        blocker = os.path.join(engine.state.detect_dir, "a.txt")
        os.makedirs(blocker)  # a directory where the file must go
        error = engine.save()
        assert error and "a.txt" in error

    def test_save_without_document(self, engine):
        engine.state.document = None
        assert engine.save() == "No image loaded"
