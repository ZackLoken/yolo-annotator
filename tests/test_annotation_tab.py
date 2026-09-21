"""Tests for yololabeler.annotation.tab, limited to what needs no display."""

import pytest

from yololabeler.annotation.document import Document, new_annotation
from yololabeler.annotation.engine import AnnotationEngine
from yololabeler.annotation.tab import AnnotateTab
from yololabeler.review.engine import QueueItem
from yololabeler.state import AppState


@pytest.fixture
def tab():
    """An AnnotateTab over a bare AppState; visible_annotations needs no
    canvas.
    """
    state = AppState()
    state._engine = AnnotationEngine(state)
    state.document = Document("a.jpg", 640, 480)
    return AnnotateTab(state)


def add(tab, kind, class_id):
    """Append an annotation of the given kind and class to the fixture's
    document.
    """
    points = (
        ((0, 0), (10, 10)) if kind == "box" else ((0, 0), (10, 0), (10, 10))
    )
    ann = new_annotation(kind, points, class_id, "tester")
    tab.app.document.add(ann)
    return ann


# ── visible_annotations ─────────────────────────────────────────────────────


class TestVisibleAnnotations:
    def test_matching_kind_and_class_is_visible(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        ann = add(tab, "box", 0)
        assert tab.visible_annotations() == [ann]

    def test_other_kind_is_hidden(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        add(tab, "polygon", 0)
        assert tab.visible_annotations() == []

    def test_other_class_is_hidden(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        tab.app._review_filter_class = 0
        add(tab, "box", 1)
        assert tab.visible_annotations() == []

    def test_all_class_filter_shows_every_class(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        tab.app._review_filter_class = "all"
        zero = add(tab, "box", 0)
        other = add(tab, "box", 1)
        assert tab.visible_annotations() == [zero, other]

    def test_all_class_filter_still_hides_the_other_kind(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        tab.app._review_filter_class = "all"
        add(tab, "polygon", 1)
        assert tab.visible_annotations() == []

    def test_selected_annotation_of_other_kind_is_visible(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        ann = add(tab, "polygon", 0)
        tab.app._selected_annotation_id = ann.id
        assert tab.visible_annotations() == [ann]

    def test_selected_annotation_of_other_class_is_visible(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        tab.app._review_filter_class = 0
        ann = add(tab, "box", 7)
        tab.app._selected_annotation_id = ann.id
        assert tab.visible_annotations() == [ann]

    def test_queue_paired_annotation_is_visible(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        ann = add(tab, "polygon", 9)
        tab.app.queue = [QueueItem("fn", None, ann, None)]
        tab.app.queue_index = 0
        assert tab.visible_annotations() == [ann]

    def test_queue_item_without_an_annotation_hides_nothing_extra(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        visible = add(tab, "box", 0)
        add(tab, "polygon", 9)
        tab.app.queue = [QueueItem("fp", None, None, None)]
        tab.app.queue_index = 0
        assert tab.visible_annotations() == [visible]

    def test_queue_index_out_of_range_is_ignored(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        ann = add(tab, "polygon", 9)
        tab.app.queue = [QueueItem("fn", None, ann, None)]
        tab.app.queue_index = 5
        assert tab.visible_annotations() == []

    def test_visibility_toggle_hides_even_the_selection(self, tab):
        tab.app.mode = "box"
        tab.app.active_class = 0
        add(tab, "box", 0)
        selected = add(tab, "polygon", 3)
        tab.app._selected_annotation_id = selected.id
        tab.app._annotation_visible = False
        assert tab.visible_annotations() == []

    def test_no_document_is_empty(self, tab):
        tab.app.document = None
        assert tab.visible_annotations() == []

    def test_draw_order_follows_the_document(self, tab):
        tab.app.mode = "polygon"
        tab.app.active_class = 0
        first = add(tab, "polygon", 0)
        selected = add(tab, "box", 4)
        last = add(tab, "polygon", 0)
        tab.app._selected_annotation_id = selected.id
        assert tab.visible_annotations() == [first, selected, last]

    def test_an_annotation_is_listed_once_when_selected_and_matching(
        self, tab
    ):
        tab.app.mode = "box"
        tab.app.active_class = 0
        ann = add(tab, "box", 0)
        tab.app._selected_annotation_id = ann.id
        assert tab.visible_annotations() == [ann]


# ── _alive ──────────────────────────────────────────────────────────────────


class TestAlive:
    def test_known_id_is_alive(self, tab):
        ann = add(tab, "box", 0)
        assert tab._alive(ann.id) is True

    def test_unknown_id_is_not_alive(self, tab):
        add(tab, "box", 0)
        assert tab._alive("no-such-id") is False

    def test_none_is_not_alive(self, tab):
        assert tab._alive(None) is False

    def test_no_document_is_not_alive(self, tab):
        ann = add(tab, "box", 0)
        tab.app.document = None
        assert tab._alive(ann.id) is False

    def test_deleted_annotation_is_not_alive(self, tab):
        ann = add(tab, "box", 0)
        tab.app._engine.delete_annotation(ann.id)
        assert tab._alive(ann.id) is False
