"""Completion guards with real completion records and no Tk window."""

from unittest.mock import Mock

import pytest

from yololabeler.annotation.document import Document, new_annotation
from yololabeler.annotation.tab import AnnotateTab
from yololabeler.gui import YoloLabeler
from yololabeler.state import AppState
from yololabeler.state_io import AnnotationStats


@pytest.fixture
def app(tmp_path, monkeypatch):
    app = YoloLabeler.__new__(YoloLabeler)
    app._state = AppState()
    app.root = Mock()
    app.canvas = Mock()
    app.images = ["one.jpg", "two.jpg", "three.jpg"]
    app.original_image = object()
    app.document = Document("one.jpg", 640, 480)
    app.document.add(new_annotation("box", ((10, 10), (50, 50)), 0, "reviewer"))
    app._current_user = "reviewer"
    app._stats_store = AnnotationStats()
    app._stats = app._stats_store.data
    app._complete_var = Mock()
    app._complete_var.get.return_value = False
    app._complete_var.set.side_effect = lambda value: setattr(
        app._complete_var.get, "return_value", value)
    app._current_model_name = Mock(return_value="detector")
    app._annotate_tab = Mock()
    app._review_panel = Mock()
    app._record_image_time = Mock()
    app._update_filter_label = Mock()
    app.update_title = Mock()
    app.filter_var = Mock()
    app.save_current = Mock(side_effect=lambda: app._stats_store.save(tmp_path / "stats.json"))
    app._rebuild_filter()
    monkeypatch.setattr("yololabeler.gui.messagebox.askyesno", Mock(return_value=True))
    return app


@pytest.mark.parametrize("target", [1, -1, 4])
@pytest.mark.parametrize("reset_filters", [True, False])
def test_confirm_marks_current_image_complete_before_navigation(app, tmp_path, target,
                                                               reset_filters):
    def check_saved_completion():
        store, moved = AnnotationStats.load(tmp_path / "stats.json")
        assert moved is None
        assert store.image_status("one.jpg") == "complete"
        completion = store.completion("one.jpg")
        assert completion["by"] == "reviewer"
        assert completion["annotation_count"] == 1
        assert completion["model"] == "detector"

    app._annotate_tab.load_image.side_effect = check_saved_completion
    assert app.go_to_image(target, reset_filters=reset_filters)
    assert app.index == target % 3
    assert "one.jpg" in app._completed_images
    app._annotate_tab.load_image.assert_called_once_with()


def test_decline_preserves_image_completion_focus_and_filters(app, monkeypatch):
    prompt = Mock(return_value=False)
    monkeypatch.setattr("yololabeler.gui.messagebox.askyesno", prompt)
    app._review_filter_type = "fp"
    app._review_status_filter = "not_reviewed"

    assert not app.go_to_image(1)

    assert app.index == 0
    assert app._stats_store.completion("one.jpg") is None
    assert app._completed_images == set()
    assert app._review_filter_type == "fp"
    assert app._review_status_filter == "not_reviewed"
    app._annotate_tab.load_image.assert_not_called()
    app._record_image_time.assert_not_called()
    app._review_panel.refresh.assert_not_called()
    assert prompt.call_args.kwargs["default"] == "no"


@pytest.mark.parametrize("target,complete", [(1, True), (0, False), (3, False)])
def test_completed_image_or_same_image_needs_no_confirmation(app, monkeypatch, target,
                                                           complete):
    prompt = Mock()
    monkeypatch.setattr("yololabeler.gui.messagebox.askyesno", prompt)
    app._complete_var.set(complete)
    assert app.go_to_image(target)
    prompt.assert_not_called()


def test_save_failure_stops_before_prompt(app, monkeypatch):
    prompt = Mock()
    monkeypatch.setattr("yololabeler.gui.messagebox.askyesno", prompt)
    app.save_current = Mock(return_value="Could not save")
    assert not app.go_to_image(1)
    prompt.assert_not_called()
    app._annotate_tab.load_image.assert_not_called()


@pytest.mark.parametrize("direction", ["next_image", "prev_image"])
def test_navigation_buttons_use_guard(app, monkeypatch, direction):
    monkeypatch.setattr("yololabeler.gui.messagebox.askyesno", Mock(return_value=False))
    tab = AnnotateTab.__new__(AnnotateTab)
    tab.app = app
    getattr(tab, direction)()
    assert app.index == 0
    app._annotate_tab.load_image.assert_not_called()


@pytest.mark.parametrize("choice", ["Complete", "Partial"])
def test_declined_filter_change_restores_filter_and_canvas(app, monkeypatch, choice):
    monkeypatch.setattr("yololabeler.gui.messagebox.askyesno", Mock(return_value=False))
    image = app.original_image
    app._on_filter_changed(choice)
    assert app.original_image is image
    assert app._active_filter == "all"
    assert app._filtered_indices == [0, 1, 2]
    app.filter_var.set.assert_called_once_with("All")
    app.canvas.delete.assert_not_called()
    app._annotate_tab.load_image.assert_not_called()


def test_completion_rebuilds_new_filter_before_loading(app):
    app._on_filter_changed("Complete")
    assert app._active_filter == "complete"
    assert app._filtered_indices == [0]
    assert app._stats_store.completion("one.jpg") is not None
    app._annotate_tab.load_image.assert_called_once_with()


def test_empty_filter_marks_complete_before_clearing_canvas(app):
    app._on_filter_changed("Partial")
    assert app._stats_store.completion("one.jpg") is not None
    assert app.original_image is None
    app.canvas.delete.assert_called_once_with("all")


def test_declined_folder_change_keeps_current_folder(app, monkeypatch):
    monkeypatch.setattr("yololabeler.gui.filedialog.askdirectory", lambda **kw: "new-folder")
    monkeypatch.setattr("yololabeler.gui.messagebox.askyesno", Mock(return_value=False))
    app._init_folder = Mock()
    app._end_session = Mock()
    app._open_folder()
    app._init_folder.assert_not_called()
    app._end_session.assert_not_called()
    app._record_image_time.assert_not_called()


def test_blind_completion_keeps_blind_provenance(app):
    app._stats_store.set_blind("one.jpg", True)
    assert app.go_to_image(1)
    completion = app._stats_store.completion("one.jpg")
    assert completion["blind"] is True
    assert completion["model"] is None
