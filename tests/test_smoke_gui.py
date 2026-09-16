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


@pytest.fixture
def poly_folder(tmp_path):
    """An image folder whose one polygon label is matched by one of two predictions.

    The GT polygon is the pixel square (192,144)-(320,240) and the first
    prediction is identical to it, so the queue is one fp (the far class-1
    polygon) and one tp.
    """
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (640, 480), "gray").save(tmp_path / name)
    s = tmp_path / "labels" / "segment"
    s.mkdir(parents=True)
    (s / "a.txt").write_text("0 0.3 0.3 0.5 0.3 0.5 0.5 0.3 0.5\n", encoding="utf-8")
    p = tmp_path / "predictions" / "segment"
    p.mkdir(parents=True)
    (p / "a.txt").write_text(
        "0 0.9 0.3 0.3 0.5 0.3 0.5 0.5 0.3 0.5\n"
        "1 0.8 0.75 0.75 0.875 0.75 0.875 0.9 0.75 0.9\n", encoding="utf-8")
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


@pytest.fixture
def poly_app(poly_folder):
    """A live YoloLabeler on the polygon folder; skipped when Tk has no display."""
    root = new_root()
    root.geometry("900x600")
    app = YoloLabeler(root)
    root.update()
    app._init_folder(str(poly_folder))
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
        # Undo load_image()'s auto-zoom onto the fp item so the GT box isn't culled off-screen.
        tab.fit_to_window()
        app.queue = []
        app._review_show_pred = False
        ann = app.document.annotations[0]
        # Undo load_image()'s auto-selected fp class so the GT box passes the class-match check.
        app._select_class_by_id(ann.class_id)
        tab.render()
        drawn = shapes_at(tab, ann.points)
        assert [c for c, _ in drawn] == [app._get_class_color(ann.class_id)]

    def test_selected_box_gets_a_white_halo(self, app):
        tab = app._annotate_tab
        tab.fit_to_window()
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

    def test_load_image_zooms_to_first_unreviewed(self, app):
        # No explicit focus_item/step call: load_image() alone must trigger the zoom.
        tab = app._annotate_tab
        tab.load_image()
        item = app.queue[app.queue_index]
        shape = item.prediction or item.annotation
        xs = [p[0] for p in shape.points]
        ys = [p[1] for p in shape.points]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        canvas_x, canvas_y = tab.image_to_canvas(cx, cy)
        # Only zoom_to_bbox, not a plain fit_to_window, centers the focused item like this.
        assert canvas_x == pytest.approx(tab.canvas.winfo_width() / 2, abs=1.0)
        assert canvas_y == pytest.approx(tab.canvas.winfo_height() / 2, abs=1.0)


class TestClassFilterMerge:
    def test_all_choice_sets_filter_and_leaves_active_class(self, app):
        before = app.active_class
        app._on_class_selected("All")
        assert app._review_filter_class == "all"
        assert app.active_class == before

    def test_specific_choice_sets_both_and_dropdown_shows_it(self, app):
        # Regression guard: _review_filter_class must be set before
        # _select_class_by_id runs, or _refresh_class_dropdown snaps back to "All".
        count = app._count_class_annotations().get(0, 0)
        choice = f"0: {app.class_names[0]} ({count})"
        app._on_class_selected(choice)
        assert app.active_class == 0
        assert app._review_filter_class == 0
        assert app.class_dropdown.get() == choice

    def test_all_filter_survives_focus_item_on_other_class(self, app):
        app._on_class_selected("All")
        other = next(i for i, q in enumerate(app.queue) if q.class_id != app.active_class)
        app._review_panel.focus_item(other)
        assert app.active_class == app.queue[other].class_id
        assert app.class_dropdown.get() == "All"


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

    def test_dragging_the_focused_vertex_refreshes_the_display_and_keeps_the_verdict(
            self, poly_app):
        app = poly_app
        panel, tab = app._review_panel, app._annotate_tab
        tp_index = next(i for i, q in enumerate(app.queue) if q.kind == "tp")
        panel.focus_item(tp_index)
        key = panel.current_item().key
        app.accept_item()
        before = dict(app.verdicts[key])
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.key == key))
        app.edit_pair()
        vx, vy = panel.current_item().annotation.points[0]
        assert panel.item_label.cget("text").startswith("TP")
        assert panel.counts_label.cget("text") == "TP 1  FP 1  FN 0"
        tab.on_button_press(click_at(tab, vx, vy))
        tab.on_move_press(click_at(tab, 10, 10))
        tab.on_button_release(click_at(tab, 10, 10))
        # The drag alone pulls the GT below the IoU threshold, and the strip says so.
        assert panel.counts_label.cget("text") == "TP 0  FP 2  FN 1"
        assert panel.item_label.cget("text").startswith("FP")
        assert app.verdicts[key] == before

    def test_drawing_an_unrelated_box_records_nothing_for_the_focused_item(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        panel.focus_item(len(app.queue) - 1)
        item = panel.current_item()
        assert item.kind == "tp"
        before = dict(app.verdicts)
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        assert len(app.document.annotations) == 2
        assert app.verdicts == before and item.key not in app.verdicts
        assert panel.current_item().key == item.key

    def test_rejecting_a_tp_does_not_select_the_deleted_annotation(self, app):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        item = panel.current_item()
        assert item.kind == "tp"
        ann_id = item.annotation.id
        app.reject_item()
        assert all(a.id != ann_id for a in app.document.annotations)
        assert app._selected_annotation_id != ann_id

    def test_accepting_an_fp_pauses_on_the_new_annotation(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        item = panel.current_item()
        assert item.kind == "fp"
        key = item.key
        app.accept_item()
        created = app.document.annotations[-1]
        assert created.source == "accepted"
        assert app._selected_annotation_id == created.id
        after = panel.current_item()
        assert after.key == key and after.kind == "tp"


# ── auto-advance under filters ──────────────────────────────────────────────

class TestAutoAdvanceFilters:
    def test_reviewed_filter_does_not_advance_on_a_re_accept(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        assert panel.current_item().kind == "fp"
        tp_key = next(q.key for q in app.queue if q.kind == "tp")
        app.accept_item()
        panel.on_status_changed("Reviewed")
        # Every item in this view has a verdict by construction, but the tp does not.
        assert app.queue and all(q.key in app.verdicts for q in app.queue)
        assert tp_key not in app.verdicts
        panel.focus_item(0)
        app.accept_item()
        assert app.images[app.index] == "a.jpg"

    def test_not_reviewed_filter_advances_after_the_last_unreviewed_item(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        assert panel.current_item().kind == "fp"
        app.accept_item()
        panel.on_status_changed("Not reviewed")
        assert [q.kind for q in app.queue] == ["tp"]
        panel.focus_item(0)
        app.accept_item()
        assert app.images[app.index] == "b.jpg"

    def test_type_filter_still_advances_when_that_type_is_done(self, app, folder):
        panel = app._review_panel
        panel.on_type_changed("FP")
        assert [q.kind for q in app.queue] == ["fp"]
        fp_key = app.queue[0].key
        panel.focus_item(0)
        app.reject_item()
        assert app.images[app.index] == "b.jpg"
        assert list(disk_verdicts(folder)) == [fp_key]

    def test_class_filter_still_advances_when_that_class_is_done(self, app, folder):
        panel = app._review_panel
        app._review_filter_class = 1
        panel.refresh(keep_focus=False)
        assert [q.kind for q in app.queue] == ["fp"]
        fp_key = app.queue[0].key
        panel.focus_item(0)
        app.reject_item()
        assert app.images[app.index] == "b.jpg"
        assert list(disk_verdicts(folder)) == [fp_key]


# ── undo/redo with no folder open ───────────────────────────────────────────

class TestUndoRedoNoFolder:
    def test_no_exception_with_no_folder_open(self):
        root = new_root()
        app = YoloLabeler(root)
        root.update()
        app.undo()
        app.redo()
        app._quit()


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

    def test_manual_navigation_resets_type_status_filters_not_class(self, app):
        app._review_filter_type = "fp"
        app._review_filter_class = 3
        app.go_to_image(1)
        assert app._review_filter_type == "all"
        assert app._review_filter_class == 3

    def test_reset_filters_false_preserves_filters(self, app):
        app._review_filter_type = "fp"
        app._review_filter_class = 3
        app._review_status_filter = "reviewed"
        app.go_to_image(1, reset_filters=False)
        assert app._review_filter_type == "fp"
        assert app._review_filter_class == 3
        assert app._review_status_filter == "reviewed"


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

        def iconphoto(self, *a, **k):
            pass

        def iconbitmap(self, *a, **k):
            pass

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
        # Undo load_image()'s auto-selected fp class so the swatch reflects class 0's color.
        app._select_class_by_id(0)
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
