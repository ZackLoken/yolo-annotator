"""Headed smoke test: build the real window and drive the canvas; skipped without a display."""

import tkinter as tk

import pytest
from PIL import Image

pytest.importorskip("customtkinter")
import customtkinter as ctk  # noqa: E402

from yololabeler.gui import YoloLabeler  # noqa: E402


def click_at(tab, ix, iy):
    """A synthetic click event on the canvas point showing image pixel (ix, iy)."""
    cx, cy = tab.image_to_canvas(ix, iy)
    return type("E", (), {"x": int(cx), "y": int(cy)})()


@pytest.fixture
def folder(tmp_path):
    """An image folder with one labelled image, one unlabelled, and predictions."""
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (640, 480), "gray").save(tmp_path / name)
    d = tmp_path / "labels" / "detect"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    p = tmp_path / "predictions" / "detect"
    p.mkdir(parents=True)
    (p / "a.txt").write_text("0 0.9 0.5 0.5 0.2 0.2\n1 0.8 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def app(folder):
    """A live YoloLabeler on that folder; skipped when Tk has no display."""
    try:
        root = ctk.CTk()
    except tk.TclError as e:
        pytest.skip(f"no Tk display: {e}")
    root.geometry("900x600")
    app = YoloLabeler(root)
    root.update()
    app._init_folder(str(folder))
    app._annotate_tab.load_image()
    root.update()
    yield app
    app._quit()


# ── annotate ────────────────────────────────────────────────────────────────

class TestAnnotate:
    def test_loads_document(self, app):
        assert app.document is not None
        assert [a.kind for a in app.document.annotations] == ["box"]
        assert app.mode == "box"

    def test_draw_delete_undo(self, app):
        tab = app._annotate_tab
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert len(app.document.annotations) == 2
        tab.undo_last()
        assert len(app.document.annotations) == 1
        tab.redo_last()
        assert len(app.document.annotations) == 2

    def test_save_and_reload(self, app):
        tab = app._annotate_tab
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert tab.save_annotations() is None
        tab.load_image()
        assert len(app.document.annotations) == 2
        assert app.document.annotations[1].author == app._current_user

    def test_select_annotation_keeps_a_box_selected(self, app):
        ann = app.document.annotations[0]
        app._annotate_tab.select_annotation(ann.id)
        assert app.mode == "box"
        assert app._selected_annotation_id == ann.id

    def test_hidden_class_is_not_a_delete_target(self, app):
        app._select_class_by_id(0)
        app.class_names[1] = "other"
        app._select_class_by_id(1)
        assert app._annotate_tab.visible_annotations() == []


# ── review panel ────────────────────────────────────────────────────────────

class TestReviewPanel:
    def test_queue_over_image_with_predictions(self, app):
        assert [q.kind for q in app.queue] == ["fp", "tp"]

    def test_step_sets_class_and_zooms(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        assert app.active_class == app.queue[0].class_id
        assert app._annotate_tab.scale != 1.0
        panel.step(1)
        assert app.queue_index == 1

    def test_threshold_filters_queue(self, app):
        app._review_panel.set_threshold(0.85)
        assert [q.kind for q in app.queue] == ["tp"]
        assert app._review.conf_threshold == 0.85

    def test_image_without_predictions_is_a_stop(self, app):
        app._annotate_tab.next_image()
        assert app.images[app.index] == "b.jpg"
        assert app.queue == [] and app.predictions == []


# ── actions ─────────────────────────────────────────────────────────────────

class TestActions:
    def test_accept_fp_promotes_and_records(self, app):
        app._review_panel.focus_item(0)
        assert app.queue[0].kind == "fp"
        app.accept_item()
        assert len(app.document.annotations) == 2
        added = app.document.annotations[-1]
        assert added.source == "accepted" and added.class_id == 1
        assert app.verdicts[added.prediction_id]["action"] == "accepted"
        assert [q.kind for q in app.queue] == ["tp", "tp"]

    def test_reject_tp_deletes_and_undo_restores_both(self, app):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        assert app.queue[app.queue_index].kind == "tp"
        key = app.queue[app.queue_index].key
        app.reject_item()
        assert app.document.annotations == [] and app.verdicts[key]["action"] == "rejected"
        app.undo()
        assert len(app.document.annotations) == 1 and key not in app.verdicts

    def test_edit_pair_selects_annotation(self, app):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        app.edit_pair()
        assert app._selected_annotation_id == app.queue[app.queue_index].annotation.id


# ── navigation and saving ───────────────────────────────────────────────────

class TestNavigation:
    def test_go_to_image_saves_first(self, app, folder):
        tab = app._annotate_tab
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert app.go_to_image(1)
        lines = (folder / "labels" / "detect" / "a.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert (folder / "state" / "annotations" / "a.json").exists()

    def test_failed_save_blocks_navigation_and_shows_banner(self, app, folder):
        import os
        broken_path = folder / "labels" / "detect" / "a.txt"
        os.remove(broken_path)
        os.makedirs(broken_path)
        app.document.annotations.clear()
        app._engine.add_box(1, 1, 30, 30)
        try:
            assert not app.go_to_image(1)
            assert app.index == 0
            assert "Could not save" in app.banner_text and "Ctrl+S" in app.banner_text
        finally:
            # Restore a normal path so the fixture's teardown quit can save cleanly
            # instead of blocking on the quit-without-saving modal.
            os.rmdir(broken_path)

    def test_banner_clears_on_key_action(self, app):
        app.show_banner("hello")
        app._key_action(lambda: None)
        assert app.banner_text is None

    def test_success_is_silent(self, app):
        app.save_now()
        assert app.banner_text is None


# ── completion and blind ────────────────────────────────────────────────────

class TestCompletion:
    def test_complete_records_and_gates(self, app, folder):
        app._complete_var.set(True)
        app._on_complete_toggled()
        rec = app._stats_store.completion("a.jpg")
        assert rec["by"] == app._current_user and rec["blind"] is False
        assert rec["annotation_count"] == 1 and rec["model"] is None
        assert app._stats_store.image_status("a.jpg") == "complete"
        app._complete_var.set(False)
        app._on_complete_toggled()
        assert app._stats_store.completion("a.jpg") is None

    def test_complete_label_counts_unreviewed(self, app):
        assert app.complete_cb.cget("text") == "Complete (2 not reviewed)"
        app._review_panel.focus_item(0)
        app.reject_item()
        assert app.complete_cb.cget("text") == "Complete (1 not reviewed)"

    def test_blind_hides_predictions_until_complete(self, app):
        app._blind_var.set(True)
        app._on_blind_toggled()
        assert app.predictions == [] and app.predictions_blind and app.queue == []
        app._complete_var.set(True)
        app._on_complete_toggled()
        assert app._stats_store.completion("a.jpg")["blind"] is True
        assert len(app.predictions) == 2 and not app.predictions_blind

    def test_model_name_from_manifest(self, app, folder):
        from yololabeler.predictions.store import write_manifest
        write_manifest(folder / "predictions", {"model": "nathan_v15"})
        app._complete_var.set(True)
        app._on_complete_toggled()
        assert app._stats_store.completion("a.jpg")["model"] == "nathan_v15"


# ── import ──────────────────────────────────────────────────────────────────

class TestImport:
    def test_run_import_reloads_predictions(self, app, folder, tmp_path):
        import json
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.json").write_text(json.dumps({"boxes": [[100, 100, 200, 200]], "scores": [0.95]}),
                                    encoding="utf-8")
        assert app._run_import(str(src), "bur_detect_json", 0, "nathan_v15") is None
        assert len(app.predictions) == 1 and app.predictions[0].confidence == 0.95
        assert "Imported predictions for 1 images" in app.banner_text
        assert app._current_model_name() == "nathan_v15"

    def test_run_import_returns_form_error(self, app, tmp_path):
        error = app._run_import(str(tmp_path), "bur_detect_json", None, "m")
        assert "class id" in error


# ── load failures ───────────────────────────────────────────────────────────

class TestLoadFailures:
    def test_bad_label_line_makes_image_read_only(self, app, folder):
        p = folder / "labels" / "detect" / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\nnope\n", encoding="utf-8")
        app._annotate_tab.load_image()
        assert app.load_errors and "line 2" in app.banner_text
        assert "will not be saved" in app.banner_text
        app._engine.add_box(1, 1, 20, 20)
        assert app.save_current() is None
        assert p.read_text(encoding="utf-8").splitlines() == ["0 0.5 0.5 0.2 0.2", "nope"]

    def test_bad_prediction_line_is_reported_not_fatal(self, app, folder):
        p = folder / "predictions" / "detect" / "a.txt"
        p.write_text("0 0.9 0.5 0.5 0.2 0.2\nbroken\n", encoding="utf-8")
        app._annotate_tab.load_image()
        assert len(app.predictions) == 1
        assert "line 2" in app.banner_text
