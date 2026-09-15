"""Headed smoke test: build the real window and drive the canvas; skipped without a display."""

import json
import tkinter as tk

import pytest
from PIL import Image

pytest.importorskip("customtkinter")
import customtkinter as ctk  # noqa: E402

from yololabeler import gui as guimod  # noqa: E402
from yololabeler.annotation.document import new_annotation  # noqa: E402
from yololabeler.gui import YoloLabeler  # noqa: E402
from yololabeler.review.layer import LayerStyle  # noqa: E402


def click_at(tab, ix, iy):
    """A synthetic click event on the canvas point showing image pixel (ix, iy)."""
    cx, cy = tab.image_to_canvas(ix, iy)
    return type("E", (), {"x": int(cx), "y": int(cy)})()


def disk_verdicts(folder, image="a.jpg"):
    """The verdicts review_stats.json actually holds on disk for one image."""
    path = folder / "state" / "review_stats.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("image", {}).get(image, {}).get("verdicts", {})


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


def new_root():
    """A fresh Tk root, skipping the calling test when there is no display."""
    try:
        return ctk.CTk()
    except tk.TclError as e:
        pytest.skip(f"no Tk display: {e}")


@pytest.fixture
def app(folder):
    """A live YoloLabeler on that folder; skipped when Tk has no display."""
    root = new_root()
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


# ── render ──────────────────────────────────────────────────────────────────

def shapes_at(tab, points):
    """(outline colour, width) for every canvas shape drawn at those image points."""
    canvas = tab.canvas
    want = [c for p in points for c in tab.image_to_canvas(*p)]
    return [(canvas.itemcget(i, "outline"), float(canvas.itemcget(i, "width")))
            for i in canvas.find_all()
            if canvas.coords(i) == pytest.approx(want, abs=0.5)]


def add_polygon(app):
    """Append a polygon annotation of the active class to the loaded document."""
    ann = new_annotation("polygon", ((100, 100), (200, 100), (200, 200)),
                         app.active_class, "tester")
    app.document.add(ann)
    return ann


class TestRenderSelection:
    def test_unselected_box_is_drawn_once_in_the_class_colour(self, app):
        tab = app._annotate_tab
        app.queue = []
        app._review_show_pred = False
        ann = app.document.annotations[0]
        tab.render()
        drawn = shapes_at(tab, ann.points)
        assert [c for c, _ in drawn] == [app._get_class_color(ann.class_id)]

    def test_selected_box_gets_a_white_halo(self, app):
        tab = app._annotate_tab
        app.queue = []
        app._review_show_pred = False
        ann = app.document.annotations[0]
        tab.select_annotation(ann.id)
        tab.render()
        drawn = dict(shapes_at(tab, ann.points))
        color = app._get_class_color(ann.class_id)
        assert set(drawn) == {"white", color}
        assert drawn["white"] == pytest.approx(drawn[color] + 2)

    def test_unselected_polygon_is_drawn_once_in_the_class_colour(self, app):
        tab = app._annotate_tab
        app.queue = []
        app._review_show_pred = False
        ann = add_polygon(app)
        tab.select_annotation(ann.id)
        tab.select_annotation(None)
        tab.render()
        drawn = shapes_at(tab, ann.points)
        assert [c for c, _ in drawn] == [app._get_class_color(ann.class_id)]

    def test_selected_polygon_gets_a_white_halo(self, app):
        tab = app._annotate_tab
        app.queue = []
        app._review_show_pred = False
        ann = add_polygon(app)
        tab.select_annotation(ann.id)
        tab.render()
        drawn = dict(shapes_at(tab, ann.points))
        color = app._get_class_color(ann.class_id)
        assert set(drawn) == {"white", color}
        assert drawn["white"] == pytest.approx(drawn[color] + 2)


class TestRenderFocus:
    def test_focused_annotation_is_drawn_only_in_the_focus_colour(self, app):
        tab = app._annotate_tab
        app._review_panel.focus_item(len(app.queue) - 1)
        ann = app.queue[app.queue_index].annotation
        assert ann is not None
        tab.render()
        assert len(tab.canvas.find_withtag("gt_focus")) == 1
        colors = [c for c, _ in shapes_at(tab, ann.points)]
        assert colors.count(LayerStyle().focused_gt_color) == 1
        assert app._get_class_color(ann.class_id) not in colors

    def test_editing_the_focused_pair_keeps_the_class_coloured_shape(self, app):
        tab = app._annotate_tab
        app._review_panel.focus_item(len(app.queue) - 1)
        ann = app.queue[app.queue_index].annotation
        app.edit_pair()
        tab.render()
        colors = [c for c, _ in shapes_at(tab, ann.points)]
        assert app._get_class_color(ann.class_id) in colors
        assert "white" in colors


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

    def test_reject_tp_deletes_and_undo_restores_both(self, app, folder):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        assert app.queue[app.queue_index].kind == "tp"
        key = app.queue[app.queue_index].key
        app.reject_item()
        assert app.document.annotations == [] and app.verdicts[key]["action"] == "rejected"
        assert disk_verdicts(folder)[key]["action"] == "rejected"
        app.undo()
        assert len(app.document.annotations) == 1 and key not in app.verdicts
        assert key not in disk_verdicts(folder)

    def test_edit_pair_selects_annotation(self, app):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        app.edit_pair()
        assert app._selected_annotation_id == app.queue[app.queue_index].annotation.id

    def test_accept_writes_the_label_file_at_once(self, app, folder):
        app._review_panel.focus_item(0)
        assert app.queue[0].kind == "fp"
        app.accept_item()
        lines = (folder / "labels" / "detect" / "a.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_reject_removes_the_annotation_from_disk_at_once(self, app, folder):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        assert app.queue[app.queue_index].kind == "tp"
        app.reject_item()
        assert not (folder / "labels" / "detect" / "a.txt").exists()


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


# ── open folder ─────────────────────────────────────────────────────────────

def second_folder(folder):
    """Create and return a sibling image folder for the Open Folder tests."""
    other = folder / "other"
    other.mkdir()
    Image.new("RGB", (640, 480), "gray").save(other / "c.jpg")
    return other


class TestOpenFolder:
    def test_read_only_image_is_not_overwritten(self, app, folder, monkeypatch):
        labels = folder / "labels" / "detect" / "a.txt"
        labels.write_text("0 0.500000 0.500000 0.200000 0.200000\nnope\n", encoding="utf-8")
        app._annotate_tab.load_image()
        assert app.load_errors
        other = second_folder(folder)
        monkeypatch.setattr(guimod.filedialog, "askdirectory", lambda **kw: str(other))
        app._open_folder()
        assert app.image_folder == str(other)
        assert labels.read_text(encoding="utf-8").splitlines() == [
            "0 0.500000 0.500000 0.200000 0.200000", "nope"]

    def test_failed_save_blocks_the_switch_and_shows_the_banner(self, app, folder, monkeypatch):
        import os
        broken = folder / "labels" / "detect" / "a.txt"
        os.remove(broken)
        os.makedirs(broken)
        app._engine.add_box(1, 1, 30, 30)
        other = second_folder(folder)
        monkeypatch.setattr(guimod.filedialog, "askdirectory", lambda **kw: str(other))
        try:
            app._open_folder()
            assert app.image_folder == str(folder)
            assert "Could not save" in app.banner_text and "Ctrl+S" in app.banner_text
        finally:
            os.rmdir(broken)

    def test_folder_without_images_reports_on_the_canvas(self, app, folder, monkeypatch):
        empty = folder / "empty"
        empty.mkdir()
        monkeypatch.setattr(guimod.filedialog, "askdirectory", lambda **kw: str(empty))
        app._open_folder()
        assert app.images == [] and app.original_image is None
        assert any("No images found" in app.canvas.itemcget(i, "text")
                   for i in app.canvas.find_all() if app.canvas.type(i) == "text")


# ── rename class ────────────────────────────────────────────────────────────

def typed_name(monkeypatch, value):
    """Make the next class dialog return value instead of opening a real modal."""
    class Dialog:
        def __init__(self, **kwargs):
            pass

        def get_input(self):
            return value

    monkeypatch.setattr(guimod.ctk, "CTkInputDialog", Dialog)


def saved_names(folder):
    """The class names classes.json holds on disk, keyed by class id."""
    data = json.loads((folder / "state" / "classes.json").read_text(encoding="utf-8"))
    return {int(k): v["name"] for k, v in data.items() if "name" in v}


class TestRenameClass:
    def test_rename_is_bound_to_a_key(self, app):
        from yololabeler.keybindings import KEY_BINDINGS
        actions = app.ACTIONS
        for b in KEY_BINDINGS:
            assert b.action in actions, b.action
        assert actions["rename_class"] == app._rename_class_dialog

    def test_rename_updates_the_registry_and_the_file(self, app, folder, monkeypatch):
        app.class_names[0] = "burr"
        app._select_class_by_id(0)
        typed_name(monkeypatch, "catkin")
        app._rename_class_dialog()
        assert app.class_names[0] == "catkin"
        assert saved_names(folder)[0] == "catkin"

    def test_an_empty_name_changes_nothing(self, app, monkeypatch):
        app.class_names[0] = "burr"
        app._select_class_by_id(0)
        typed_name(monkeypatch, "   ")
        app._rename_class_dialog()
        assert app.class_names[0] == "burr"

    def test_a_name_another_class_holds_is_refused(self, app, monkeypatch):
        app.class_names[0] = "burr"
        app.class_names[1] = "bud"
        app._select_class_by_id(0)
        typed_name(monkeypatch, "BUD")
        app._rename_class_dialog()
        assert app.class_names == {0: "burr", 1: "bud"}
        assert "Class 1 is already named" in app.banner_text

    def test_renaming_to_its_own_name_is_allowed(self, app, monkeypatch):
        app.class_names[0] = "burr"
        app._select_class_by_id(0)
        typed_name(monkeypatch, "Burr")
        app._rename_class_dialog()
        assert app.class_names[0] == "Burr"

    def test_an_unnamed_active_class_is_a_no_op(self, app, monkeypatch):
        app.class_names.clear()
        typed_name(monkeypatch, "catkin")
        app._rename_class_dialog()
        assert app.class_names == {}


# ── state files ─────────────────────────────────────────────────────────────

class TestStateFiles:
    def test_corrupt_classes_json_is_quarantined_not_overwritten(self, app, folder):
        path = folder / "state" / "classes.json"
        path.write_text("{not json", encoding="utf-8")
        app._init_folder(str(folder))
        moved = list((folder / "state").glob("classes.json.corrupt-*"))
        assert len(moved) == 1
        assert moved[0].read_text(encoding="utf-8") == "{not json"
        assert "classes.json could not be read" in app.banner_text

    def test_delete_backs_up_the_original_labels(self, app, folder):
        original = (folder / "labels" / "detect" / "a.txt").read_text(encoding="utf-8")
        app._engine.push_undo()
        app._engine.delete_annotation(app.document.annotations[0].id)
        assert app.save_current() is None
        backup = folder / "labels" / "detect" / ".original" / "a.txt"
        assert backup.read_text(encoding="utf-8") == original
        assert not (folder / "labels" / "detect" / "a.txt").exists()

    def test_opening_a_folder_repaints_the_colour_swatch(self, app, folder):
        path = folder / "state" / "classes.json"
        path.write_text(json.dumps({"0": {"name": "burr", "color": "#123456"}}),
                        encoding="utf-8")
        app._init_folder(str(folder))
        assert app.color_btn.cget("bg") == "#123456"

    def test_colour_without_a_name_survives_a_reload(self, app, folder):
        app.class_names = {}
        app.class_colors = {0: "#123456"}
        app._save_classes_file()
        app._init_folder(str(folder))
        assert app.class_colors[0] == "#123456"
        assert 0 not in app.class_names

    def test_construction_writes_no_classes_json_to_the_cwd(self, folder, monkeypatch):
        cwd = folder / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        root = new_root()
        app = YoloLabeler(root, image_folder=str(folder))
        root.update()
        assert list(cwd.iterdir()) == []
        assert (folder / "state" / "classes.json").exists()
        app._quit()


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

    def test_unreadable_image_is_skipped_and_reported(self, app, folder):
        (folder / "aa_bad.jpg").write_bytes(b"not an image")
        app._init_folder(str(folder))
        app.index = app.images.index("aa_bad.jpg")
        app._annotate_tab.load_image()
        assert app.images[app.index] == "b.jpg"
        assert "could not be opened" in app.banner_text

    def test_bad_prediction_line_is_reported_not_fatal(self, app, folder):
        p = folder / "predictions" / "detect" / "a.txt"
        p.write_text("0 0.9 0.5 0.5 0.2 0.2\nbroken\n", encoding="utf-8")
        app._annotate_tab.load_image()
        assert len(app.predictions) == 1
        assert "line 2" in app.banner_text

    def test_bad_label_and_prediction_lines_together(self, app, folder):
        labels_path = folder / "labels" / "detect" / "a.txt"
        preds_path = folder / "predictions" / "detect" / "a.txt"
        labels_path.write_text("0 0.5 0.5 0.2 0.2\nnope\n", encoding="utf-8")
        preds_path.write_text("0 0.9 0.5 0.5 0.2 0.2\nbroken\n", encoding="utf-8")
        app._annotate_tab.load_image()
        assert len(app.load_errors) == 1 and "labels" in app.load_errors[0]
        assert len(app.predictions_rejected) == 1 and "predictions" in app.predictions_rejected[0]
        assert app.load_errors[0] in app.banner_text
        assert app.predictions_rejected[0] in app.banner_text
        assert app.save_current() is None


# ── acceptance scenarios ────────────────────────────────────────────────────

class TestAcceptance:
    def test_delete_quit_reopen_stays_deleted(self, folder):
        root = new_root()
        app = YoloLabeler(root)
        root.update()
        app._init_folder(str(folder))
        app._annotate_tab.load_image()
        app._engine.push_undo()
        app._engine.delete_annotation(app.document.annotations[0].id)
        app._quit()
        root = new_root()
        app = YoloLabeler(root)
        root.update()
        app._init_folder(str(folder))
        app._annotate_tab.load_image()
        assert app.document.annotations == []
        app._quit()

    def test_draw_step_navigate_return(self, app):
        tab = app._annotate_tab
        tab.on_button_press(click_at(tab, 100, 100))
        tab.on_button_release(click_at(tab, 160, 160))
        app._review_panel.step(1)
        assert app.go_to_image(1) and app.go_to_image(0)
        assert len(app.document.annotations) == 2

    def test_resize_refits(self, app):
        before = app._annotate_tab.scale
        app.root.geometry("700x500")
        app.root.update()
        app._annotate_tab._finalize_resize()
        assert app._annotate_tab.scale != before or app._annotate_tab.offset_x != 0

    def test_folder_without_predictions_lists_every_image(self, folder):
        import shutil
        shutil.rmtree(folder / "predictions")
        root = new_root()
        app = YoloLabeler(root)
        root.update()
        app._init_folder(str(folder))
        app._annotate_tab.load_image()
        assert app._filtered_indices == [0, 1] and app.queue == []
        app._quit()

    def test_help_matches_binding_table(self, app):
        from yololabeler.keybindings import KEY_BINDINGS
        app._annotate_tab.toggle_help()
        texts = " ".join(app.canvas.itemcget(i, "text") for i in app.canvas.find_all()
                         if app.canvas.type(i) == "text")
        for b in KEY_BINDINGS:
            if b.when == "always":
                assert b.label in texts
