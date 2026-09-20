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
from yololabeler.review.engine import build_queue  # noqa: E402
from yololabeler.review.layer import SELECTION_COLOR, STATUS_COLORS  # noqa: E402


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


def disk_img_status(folder, image="a.jpg"):
    """The img_status review_stats.json actually holds on disk for one image."""
    path = folder / "state" / "review_stats.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("image", {}).get(image, {}).get("img_status")


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


def drop_deferred_callbacks(root):
    """Cancel every pending root.after timer so none of them fires part-way through a test.

    The app arms a 100ms welcome screen when it is built with no folder, and the
    fixtures build it that way before opening one. Left pending, it fires after
    the image has loaded and wipes the canvas. Idle callbacks are left alone
    because the canvas redraw throttle only re-arms once its own idle callback
    has run.
    """
    for after_id in root.tk.splitlist(root.tk.call("after", "info")):
        if root.tk.splitlist(root.tk.call("after", "info", after_id))[-1] == "timer":
            root.after_cancel(after_id)


@pytest.fixture
def app(folder):
    """A live YoloLabeler on that folder; skipped when Tk has no display."""
    root = new_root()
    root.geometry("900x600")
    app = YoloLabeler(root)
    drop_deferred_callbacks(root)
    root.update()
    app._init_folder(str(folder))
    app._annotate_tab.load_image()
    root.update()
    drop_deferred_callbacks(root)
    yield app
    app._quit()


@pytest.fixture
def poly_app(poly_folder):
    """A live YoloLabeler on the polygon folder; skipped when Tk has no display."""
    root = new_root()
    root.geometry("900x600")
    app = YoloLabeler(root)
    drop_deferred_callbacks(root)
    root.update()
    app._init_folder(str(poly_folder))
    app._annotate_tab.load_image()
    root.update()
    drop_deferred_callbacks(root)
    yield app
    app._quit()


# ── annotate ────────────────────────────────────────────────────────────────

class TestAnnotate:
    def test_load_image_renders_once(self, app):
        tab = app._annotate_tab
        renders = []
        tab.render = lambda: renders.append(1)
        tab.load_image()
        app.root.update()
        assert len(renders) == 1

    def test_renaming_the_class_redraws_its_labels_at_once(self, app, monkeypatch):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        monkeypatch.setattr(app, "_class_name_dialog", lambda text, title: "hazelnut")
        app._rename_class_dialog()
        labels = [tab.canvas.itemcget(i, "text") for i in tab.canvas.find_all()
                  if tab.canvas.type(i) == "text"]
        assert any(t.startswith("0: hazelnut") for t in labels)

    def test_loads_document(self, app):
        assert app.document is not None
        assert [a.kind for a in app.document.annotations] == ["box"]
        assert app.mode == "box"

    def test_draw_delete_undo(self, app):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert len(app.document.annotations) == 2
        tab.undo_last()
        assert len(app.document.annotations) == 1
        tab.redo_last()
        assert len(app.document.annotations) == 2

    def test_save_and_reload(self, app):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
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

    def test_image_timer_starts_on_load(self, app):
        app._image_start_time = None
        app._annotate_tab.load_image()
        assert app._image_start_time is not None

    def test_hidden_class_is_not_a_delete_target(self, app):
        app._select_class_for_filter_and_draw(0)
        app.class_names[1] = "other"
        app._select_class_for_filter_and_draw(1)
        assert app._annotate_tab.visible_annotations() == []

    def test_the_all_class_filter_shows_every_class(self, app):
        app.class_names[1] = "other"
        app.document.add(new_annotation("box", ((10, 10), (60, 60)), 1, "tester"))
        app._select_class_for_filter_and_draw(0)
        assert [a.class_id for a in app._annotate_tab.visible_annotations()] == [0]
        app._on_class_selected("All")
        assert [a.class_id for a in app._annotate_tab.visible_annotations()] == [0, 1]

    def test_picking_a_class_clears_the_draw_prompt_banner(self, app):
        app.show_banner("Select a class before drawing.")
        app._on_class_selected(f"0: {app.class_names[0]} (1)")
        assert app.banner_text is None


# ── window resize keeps the zoom ────────────────────────────────────────────

def resize_event(width, height):
    """A synthetic canvas Configure event."""
    return type("E", (), {"width": width, "height": height})()


class TestResizeKeepsZoom:
    def test_resizing_keeps_the_scale_and_recentres_the_view(self, app):
        tab = app._annotate_tab
        tab._on_canvas_configure(resize_event(800, 500))
        tab._zoom_step(400, 250, 1)
        tab._zoom_step(400, 250, 1)
        scale, ox, oy = tab.scale, tab.offset_x, tab.offset_y
        tab._on_canvas_configure(resize_event(1000, 700))
        assert tab.scale == scale
        assert (tab.offset_x, tab.offset_y) == (ox + 100, oy + 100)
        tab._finalize_resize()
        assert tab.scale == scale

    def test_a_fit_guessed_before_the_canvas_was_mapped_is_redone_once(self, app):
        tab = app._annotate_tab
        tab._on_canvas_configure(resize_event(800, 500))
        tab._fit_pending = True
        tab.scale, tab.offset_x, tab.offset_y = 4.0, -999.0, -999.0
        tab._on_canvas_configure(resize_event(800, 500))
        assert tab.scale < 4.0 and tab.offset_x >= 0 and tab.offset_y >= 0
        assert tab._fit_pending is False
        scale = tab.scale
        tab._on_canvas_configure(resize_event(900, 500))
        assert tab.scale == scale

    def test_the_f_key_still_fits(self, app):
        tab = app._annotate_tab
        tab._on_canvas_configure(resize_event(800, 500))
        tab._zoom_step(400, 250, 1)
        zoomed = tab.scale
        tab.fit_to_window()
        assert tab.scale != zoomed


# ── box editing ─────────────────────────────────────────────────────────────

def box_setup(app):
    """The tab and the fixture's GT box, with its class chosen and the image in view."""
    app._select_class_for_filter_and_draw(0)
    tab = app._annotate_tab
    tab.fit_to_window()
    return tab, app.document.annotations[0]


def image_point(tab, ix, iy):
    """The image point a synthetic click at (ix, iy) lands on after pixel rounding."""
    event = click_at(tab, ix, iy)
    return tab.canvas_to_image(event.x, event.y)


def flat(points):
    """A point pair flattened for a single approx comparison."""
    return [c for p in points for c in p]


class TestBoxEditing:
    def test_dragging_a_corner_leaves_the_opposite_corner_fixed(self, app):
        tab, ann = box_setup(app)
        (x1, y1), (x2, y2) = ann.points
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, x1, y1))
        assert app._box_edit_mode == "resize"
        assert app._box_edit_anchor == (x2, y2)
        tab.on_move_press(click_at(tab, x1 + 40, y1 + 30))
        tab.on_button_release(click_at(tab, x1 + 40, y1 + 30))
        moved, fixed = app.document.get(ann.id).points
        assert fixed == (x2, y2)
        assert moved == pytest.approx(image_point(tab, x1 + 40, y1 + 30))
        assert app._box_edit_mode is None and app._box_edit_dirty is False

    def test_dragging_the_outline_shifts_both_points_by_the_same_delta(self, app):
        tab, ann = box_setup(app)
        (x1, y1), (x2, y2) = ann.points
        cx, cy = top_edge_mid(ann)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, cx, cy))
        assert app._box_edit_mode == "move"
        sx, sy = image_point(tab, cx, cy)
        tx, ty = image_point(tab, cx + 60, cy - 20)
        tab.on_move_press(click_at(tab, cx + 60, cy - 20))
        tab.on_button_release(click_at(tab, cx + 60, cy - 20))
        dx, dy = tx - sx, ty - sy
        assert flat(app.document.get(ann.id).points) == pytest.approx(
            [x1 + dx, y1 + dy, x2 + dx, y2 + dy])

    def test_dragging_the_outline_past_an_edge_clamps_it_inside_the_image(self, app):
        tab, ann = box_setup(app)
        (x1, y1), (x2, y2) = ann.points
        cx, cy = top_edge_mid(ann)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, cx, cy))
        tab.on_move_press(click_at(tab, x2 + 400, y2 + 400))
        tab.on_button_release(click_at(tab, x2 + 400, y2 + 400))
        assert flat(app.document.get(ann.id).points) == pytest.approx(
            [app.img_width - (x2 - x1), app.img_height - (y2 - y1),
             app.img_width, app.img_height])

    def test_shift_press_in_box_mode_is_an_ordinary_press(self, app):
        tab = app._annotate_tab
        ann = app.document.annotations[0]
        app._select_class_for_filter_and_draw(0)
        tab.scale, tab.offset_x, tab.offset_y = 1.0, 0.0, 0.0
        tab.select_annotation(ann.id)
        cx, cy = box_center(ann)
        tab.on_shift_press(click_at(tab, cx, cy))
        assert app._selected_annotation_id is None
        assert app.start_x is not None

    def test_a_box_wider_than_the_image_still_drags(self, app):
        tab = app._annotate_tab
        ann_id = app.document.annotations[0].id
        app._engine.set_points(ann_id, ((-20, -10), (700, 500)))
        ann = app.document.get(ann_id)
        tab.scale, tab.offset_x, tab.offset_y = 0.5, 100.0, 100.0
        drag_box_to(tab, ann, 345, 250)
        (x1, y1), (x2, y2) = app.document.get(ann.id).points
        # A small drag moves both corners by the same few pixels; it used to pin x1 at 0.
        assert 0 < x1 + 20 < 10 and x2 - 700 == pytest.approx(x1 + 20)
        assert 0 < y1 + 10 < 10 and y2 - 500 == pytest.approx(y1 + 10)

    def test_a_degenerate_resize_rolls_back_and_cannot_be_redone(self, app):
        tab, ann = box_setup(app)
        before = ann.points
        (x1, y1), (x2, y2) = before
        tab.select_annotation(ann.id)
        undo_before, redo_before = len(app._undo_stack), len(app._redo_stack)
        tab.on_button_press(click_at(tab, x1, y1))
        tab.on_move_press(click_at(tab, x2 - 1, y2 - 1))
        (dx1, dy1), (dx2, dy2) = app.document.get(ann.id).points
        assert dx2 - dx1 < 3 and dy2 - dy1 < 3
        tab.on_button_release(click_at(tab, x2 - 1, y2 - 1))
        assert app.document.get(ann.id).points == before
        assert len(app._undo_stack) == undo_before
        assert len(app._redo_stack) == redo_before
        app.redo()
        assert app.document.get(ann.id).points == before

    def test_a_click_with_no_drag_pushes_no_undo_and_keeps_the_redo_stack(self, app):
        tab, ann = box_setup(app)
        tab.on_button_press(click_at(tab, 40, 40))
        tab.on_button_release(click_at(tab, 140, 140))
        tab.undo_last()
        assert (len(app._undo_stack), len(app._redo_stack)) == (0, 1)
        tab.select_annotation(ann.id)
        before = app.document.get(ann.id).points
        (x1, y1), _ = before
        tab.on_button_press(click_at(tab, x1, y1))
        assert app._box_edit_mode == "resize"
        tab.on_button_release(click_at(tab, x1, y1))
        assert (len(app._undo_stack), len(app._redo_stack)) == (0, 1)
        assert app.document.get(ann.id).points == before

    def test_clicking_another_box_selects_it_without_starting_a_draw(self, app):
        tab, ann = box_setup(app)
        tab.on_button_press(click_at(tab, 40, 40))
        tab.on_button_release(click_at(tab, 140, 140))
        other = app.document.annotations[-1]
        tab.select_annotation(other.id)
        count = len(app.document.annotations)
        tab.on_button_press(click_at(tab, *top_edge_mid(ann)))
        assert app._selected_annotation_id == ann.id
        assert app.rect is None
        tab.on_button_release(click_at(tab, *top_edge_mid(ann)))
        assert len(app.document.annotations) == count
        assert app._box_edit_mode is None

    def test_clicking_an_outline_with_nothing_selected_selects_it(self, app):
        tab, ann = box_setup(app)
        assert app._selected_annotation_id is None and app.mode == "box"
        count = len(app.document.annotations)
        tab.on_button_press(click_at(tab, *top_edge_mid(ann)))
        tab.on_button_release(click_at(tab, *top_edge_mid(ann)))
        assert app._selected_annotation_id == ann.id
        assert app.rect is None
        assert len(app.document.annotations) == count

    def test_clicking_inside_a_box_does_not_select_it(self, app):
        tab, ann = box_setup(app)
        tab.on_button_press(click_at(tab, *box_center(ann)))
        assert app._selected_annotation_id is None
        assert app.rect is not None

    def test_dragging_inside_the_selected_box_draws_a_new_box(self, app):
        tab, ann = box_setup(app)
        tab.select_annotation(ann.id)
        before = ann.points
        count = len(app.document.annotations)
        cx, cy = box_center(ann)
        tab.on_button_press(click_at(tab, cx - 20, cy - 20))
        assert app._box_edit_mode is None and app._selected_annotation_id is None
        tab.on_move_press(click_at(tab, cx + 20, cy + 20))
        tab.on_button_release(click_at(tab, cx + 20, cy + 20))
        assert len(app.document.annotations) == count + 1
        assert app.document.get(ann.id).points == before

    def test_dragging_an_unselected_outline_selects_and_moves_it(self, app):
        tab, ann = box_setup(app)
        (x1, y1), (x2, y2) = ann.points
        sx, sy = image_point(tab, *top_edge_mid(ann))
        tab.on_button_press(click_at(tab, *top_edge_mid(ann)))
        tx, ty = image_point(tab, x1 + 30, y1 + 10)
        tab.on_move_press(click_at(tab, x1 + 30, y1 + 10))
        tab.on_button_release(click_at(tab, x1 + 30, y1 + 10))
        dx, dy = tx - sx, ty - sy
        assert app._selected_annotation_id == ann.id
        assert flat(app.document.get(ann.id).points) == pytest.approx(
            [x1 + dx, y1 + dy, x2 + dx, y2 + dy])

    def test_hiding_labels_lets_you_draw_over_an_already_selected_box(self, app):
        tab, ann = box_setup(app)
        tab.select_annotation(ann.id)
        (x1, y1), (x2, y2) = ann.points
        count = len(app.document.annotations)
        app._annotation_visible = False
        tab.on_button_press(click_at(tab, x1, y1))
        assert app._box_edit_mode is None
        assert app.rect is not None
        tab.on_move_press(click_at(tab, x1 + 40, y1 + 40))
        tab.on_button_release(click_at(tab, x1 + 40, y1 + 40))
        assert len(app.document.annotations) == count + 1

    def test_switching_mode_clears_an_in_progress_box_drag(self, app):
        tab, ann = box_setup(app)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, *ann.points[0]))
        assert app._box_edit_mode == "resize"
        app._set_mode("polygon")
        app._set_mode("box")
        assert app._box_edit_mode is None

    def test_loading_an_image_clears_an_in_progress_box_drag(self, app):
        tab, ann = box_setup(app)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, *ann.points[0]))
        assert app._box_edit_mode == "resize"
        tab.load_image()
        assert app._box_edit_mode is None

    def test_undo_clears_an_in_progress_box_drag(self, app):
        tab, ann = box_setup(app)
        tab.on_button_press(click_at(tab, 40, 40))
        tab.on_button_release(click_at(tab, 140, 140))
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, *ann.points[0]))
        assert app._box_edit_mode == "resize"
        tab.undo_last()
        assert app._box_edit_mode is None


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
        # No review context at all, which is what puts an annotation in its class colour.
        app.shape_statuses = None
        app._review_show_pred = False
        ann = app.document.annotations[0]
        # Undo load_image()'s auto-selected fp class so the GT box passes the class-match check.
        app._select_class_by_id(ann.class_id)
        tab.render()
        drawn = shapes_at(tab, ann.points)
        assert [c for c, _ in drawn] == [app._get_class_color(ann.class_id)]

    def test_selected_box_is_drawn_in_the_selection_blue(self, app):
        tab = app._annotate_tab
        tab.fit_to_window()
        app.queue = []
        app._review_show_pred = False
        ann = app.document.annotations[0]
        tab.select_annotation(ann.id)
        tab.render()
        assert [c for c, _ in shapes_at(tab, ann.points)] == [SELECTION_COLOR]

    def test_selected_box_gets_four_corner_handles(self, app):
        tab = app._annotate_tab
        tab.fit_to_window()
        app.queue = []
        app._review_show_pred = False
        ann = app.document.annotations[0]
        (x1, y1), (x2, y2) = ann.points
        cx1, cy1 = tab.image_to_canvas(x1, y1)
        cx2, cy2 = tab.image_to_canvas(x2, y2)
        corners = [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]
        box_w = abs(cx2 - cx1)

        def corner_handles():
            canvas = tab.canvas
            found = []
            for i in canvas.find_all():
                if canvas.type(i) != "rectangle":
                    continue
                rx0, ry0, rx1, ry1 = canvas.coords(i)
                if abs(rx1 - rx0) >= box_w or abs(ry1 - ry0) >= box_w:
                    continue  # the full-box outline/halo rectangles, not a handle
                center = ((rx0 + rx1) / 2, (ry0 + ry1) / 2)
                if any(center == pytest.approx(corner, abs=0.5) for corner in corners):
                    found.append(i)
            return found

        tab.select_annotation(None)
        tab.render()
        assert corner_handles() == []

        tab.select_annotation(ann.id)
        tab.render()
        assert len(corner_handles()) == 4

    def test_hovering_a_box_outline_shows_its_corner_handles(self, app):
        tab = app._annotate_tab
        tab.fit_to_window()
        app.queue = []
        app._review_show_pred = False
        ann = app.document.annotations[0]
        tab._on_motion(motion_at(tab, *top_edge_mid(ann)))
        assert app._hovered_annotation_id == ann.id
        tab.render()
        (x1, y1), (x2, y2) = ann.points
        corners = [tab.image_to_canvas(x, y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        canvas = tab.canvas
        handles = [i for i in canvas.find_all() if canvas.type(i) == "rectangle"
                   and canvas.itemcget(i, "fill") == "white"
                   and any(((canvas.coords(i)[0] + canvas.coords(i)[2]) / 2,
                            (canvas.coords(i)[1] + canvas.coords(i)[3]) / 2)
                           == pytest.approx(c, abs=0.5) for c in corners)]
        assert len(handles) == 4
        # Hover checks are throttled to ~60 fps; a fast machine lands the next motion inside it.
        tab._motion_last_time = 0.0
        tab._on_motion(motion_at(tab, 5, 5))
        assert app._hovered_annotation_id is None

    def test_polygon_label_sits_at_the_top_left_of_its_bounds(self, app):
        tab = app._annotate_tab
        app.queue = []
        app._review_show_pred = False
        app._select_class_for_filter_and_draw(0)
        app._set_mode("polygon")
        tab.scale, tab.offset_x, tab.offset_y = 1.0, 0.0, 0.0
        ann = new_annotation("polygon", ((200, 200), (100, 100), (200, 100)), 0, "tester")
        app.document.add(ann)
        tab.render()
        canvas = tab.canvas
        label = f"0: {app.class_names[0]}"
        spots = {tuple(canvas.coords(i)) for i in canvas.find_all()
                 if canvas.type(i) == "text" and canvas.itemcget(i, "text") == label}
        assert (102.0, 98.0) in spots

    def test_unselected_polygon_is_drawn_once_in_the_class_colour(self, app):
        tab = app._annotate_tab
        app.queue = []
        app.shape_statuses = None
        app._review_show_pred = False
        ann = add_polygon(app)
        tab.select_annotation(ann.id)
        tab.select_annotation(None)
        tab.render()
        drawn = shapes_at(tab, ann.points)
        assert [c for c, _ in drawn] == [app._get_class_color(ann.class_id)]

    def test_selected_polygon_is_drawn_in_the_selection_blue(self, app):
        tab = app._annotate_tab
        app.queue = []
        app._review_show_pred = False
        ann = add_polygon(app)
        tab.select_annotation(ann.id)
        tab.render()
        assert [c for c, _ in shapes_at(tab, ann.points)] == [SELECTION_COLOR]

    def test_selected_polygon_vertices_are_blue_with_a_white_rim(self, app):
        tab = app._annotate_tab
        app.queue = []
        app._review_show_pred = False
        ann = add_polygon(app)
        tab.select_annotation(ann.id)
        tab.render()
        canvas = tab.canvas
        ovals = [i for i in canvas.find_all() if canvas.type(i) == "oval"
                 and canvas.itemcget(i, "fill") == SELECTION_COLOR]
        assert len(ovals) == len(ann.points)
        assert all(canvas.itemcget(i, "outline") == "white" for i in ovals)


def polygon_setup(app, scale=1.0):
    """Polygon mode, class 0 active, an unshifted view at scale, and one polygon added."""
    tab = app._annotate_tab
    app._select_class_for_filter_and_draw(0)
    app._set_mode("polygon")
    tab.scale, tab.offset_x, tab.offset_y = scale, 0.0, 0.0
    return tab, add_polygon(app)


def motion_at(tab, ix, iy):
    """A synthetic pointer-motion event over image pixel (ix, iy)."""
    cx, cy = tab.image_to_canvas(ix, iy)
    return type("E", (), {"x": cx, "y": cy})()


class TestPolygonInteraction:
    def test_switching_to_polygon_mode_drops_a_selected_box(self, app):
        tab = app._annotate_tab
        box = app.document.annotations[0]
        tab.select_annotation(box.id)
        app._set_mode("polygon")
        assert app._selected_annotation_id is None
        # A press on the box's old edge in polygon mode must not edit the box.
        tab.scale, tab.offset_x, tab.offset_y = 1.0, 0.0, 0.0
        tab.on_button_press(click_at(tab, *top_edge_mid(box)))
        tab.on_button_release(click_at(tab, *top_edge_mid(box)))
        assert len(app.document.get(box.id).points) == 2

    def test_undoing_the_only_pending_vertex_keeps_the_last_closed_polygon(self, app):
        tab, ann = polygon_setup(app)
        tab.on_button_press(click_at(tab, 300, 300))
        tab.on_button_press(click_at(tab, 400, 300))
        tab.on_button_press(click_at(tab, 400, 400))
        tab._on_double_click(click_at(tab, 400, 400))
        closed = app.document.annotations[-1]
        assert closed.id != ann.id and app.current_polygon == []
        tab.on_button_press(click_at(tab, 50, 50))
        assert app.current_polygon == [(50, 50)]
        tab.undo_last()
        assert app.current_polygon == []
        assert closed in app.document.annotations
        tab.redo_last()
        assert app.current_polygon == [(50, 50)]

    def test_an_abandoned_polygon_leaves_nothing_to_redo(self, app):
        tab, _ = polygon_setup(app)
        tab.on_button_press(click_at(tab, 300, 300))
        tab.on_button_press(click_at(tab, 400, 300))
        tab.undo_last()
        assert app._vertex_redo_stack == [(400, 300)]
        app._on_escape()
        tab.on_button_press(click_at(tab, 50, 50))
        tab.redo_last()
        assert app.current_polygon == [(50, 50)]

    def test_a_click_inside_a_polygon_starts_a_new_one(self, app):
        tab, ann = polygon_setup(app)
        tab.on_button_press(click_at(tab, 170, 130))
        assert app._selected_annotation_id is None
        assert app.current_polygon == [(170, 130)]

    def test_a_click_on_a_polygon_outline_selects_it(self, app):
        tab, ann = polygon_setup(app)
        tab.on_button_press(click_at(tab, 150, 100))
        assert app._selected_annotation_id == ann.id
        assert app.current_polygon == []

    def test_a_right_click_on_a_polygon_outline_deletes_it_and_inside_does_not(self, app):
        tab, ann = polygon_setup(app)
        tab.on_right_click(click_at(tab, 170, 130))
        assert any(a.id == ann.id for a in app.document.annotations)
        tab.on_right_click(click_at(tab, 150, 100))
        assert all(a.id != ann.id for a in app.document.annotations)

    def test_dragging_a_vertex_onto_the_opposite_edge_rolls_back(self, app):
        tab, ann = polygon_setup(app)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, 200, 100))
        tab.on_move_press(click_at(tab, 150, 150))
        tab.on_button_release(click_at(tab, 150, 150))
        assert app.document.get(ann.id).points == ann.points
        assert app._redo_stack == []

    def test_a_flat_polygon_is_discarded_with_a_banner(self, app):
        tab, _ = polygon_setup(app)
        for ix, iy in ((300, 300), (350, 350), (400, 400)):
            tab.on_button_press(click_at(tab, ix, iy))
        tab._on_double_click(click_at(tab, 400, 400))
        assert len(app.document.annotations) == 2
        assert app.current_polygon == [] and "no area" in app.banner_text

    def test_dragging_an_outline_moves_the_polygon_whole(self, app):
        tab, ann = polygon_setup(app)
        tab.on_button_press(click_at(tab, 150, 100))
        assert app._selected_annotation_id == ann.id
        tab.on_move_press(click_at(tab, 160, 120))
        tab.on_button_release(click_at(tab, 160, 120))
        moved = app.document.get(ann.id).points
        assert moved == tuple((x + 10, y + 20) for x, y in ann.points)
        assert len(app._undo_stack) == 1

    def test_dragging_the_selected_polygons_edge_moves_it_rather_than_inserting(self, app):
        tab, ann = polygon_setup(app)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, 150, 100))
        tab.on_move_press(click_at(tab, 150, 110))
        tab.on_button_release(click_at(tab, 150, 110))
        moved = app.document.get(ann.id).points
        assert len(moved) == 3
        assert moved == tuple((x, y + 10) for x, y in ann.points)

    def test_a_click_on_an_outline_without_moving_only_selects(self, app):
        tab, ann = polygon_setup(app)
        tab.on_button_press(click_at(tab, 150, 100))
        tab.on_button_release(click_at(tab, 150, 100))
        assert app._selected_annotation_id == ann.id
        assert app.document.get(ann.id).points == ann.points
        assert app._undo_stack == []

    def test_shift_click_on_the_selected_polygons_edge_inserts_a_vertex(self, app):
        tab, ann = polygon_setup(app)
        tab.select_annotation(ann.id)
        tab.on_shift_press(click_at(tab, 150, 100))
        tab.on_move_press(click_at(tab, 150, 80))
        tab.on_button_release(click_at(tab, 150, 80))
        pts = app.document.get(ann.id).points
        assert len(pts) == 4
        assert pts[1] == (150, 80)
        assert len(app._undo_stack) == 1

    def test_shift_press_off_the_selected_polygon_is_an_ordinary_press(self, app):
        tab, ann = polygon_setup(app)
        tab.select_annotation(ann.id)
        tab.on_shift_press(click_at(tab, 400, 400))
        assert tab._shape_move is None
        assert app._selected_annotation_id is None

    def test_a_click_on_an_unselected_polygons_vertex_starts_a_polygon_there(self, app):
        tab, ann = polygon_setup(app)
        tab.on_button_press(click_at(tab, 203, 98))
        assert app._selected_annotation_id is None
        assert app.current_polygon == [(200, 100)]

    def test_alt_click_on_a_vertex_selects_its_polygon(self, app):
        tab, ann = polygon_setup(app)
        tab.on_alt_press(click_at(tab, 203, 98))
        assert app._selected_annotation_id == ann.id
        assert app.current_polygon == []

    def test_the_help_overlay_lists_alt_and_shift_in_polygon_mode(self, app):
        tab, _ = polygon_setup(app)
        app.show_help = True
        tab.render()
        canvas = tab.canvas
        texts = [canvas.itemcget(i, "text") for i in canvas.find_all()
                 if canvas.type(i) == "text"]
        assert any(t.startswith("  Alt+click a vertex") for t in texts)
        assert any(t.startswith("  Shift+click an edge") for t in texts)

    def test_motion_with_alt_held_shows_the_select_cursor(self, app):
        tab, _ = polygon_setup(app)
        from yololabeler.annotation.tab import ALT_STATE_MASK
        held = type("E", (), {"x": 300.0, "y": 300.0, "state": ALT_STATE_MASK})()
        tab._on_motion(held)
        assert tab.canvas.cget("cursor") == "arrow"
        tab._on_motion(motion_at(tab, 310, 300))
        assert tab.canvas.cget("cursor") == "cross"

    def test_holding_alt_shows_the_select_cursor_in_polygon_mode(self, app):
        tab, _ = polygon_setup(app)
        assert tab.canvas.cget("cursor") == "cross"
        tab._on_alt_down()
        assert tab.canvas.cget("cursor") == "arrow"
        tab._on_alt_up()
        assert tab.canvas.cget("cursor") == "cross"

    def test_alt_leaves_a_drag_cursor_and_box_mode_alone(self, app):
        tab, ann = polygon_setup(app)
        tab.canvas.config(cursor="fleur")
        tab._on_alt_down()
        assert tab.canvas.cget("cursor") == "fleur"
        tab._on_alt_up()
        assert tab.canvas.cget("cursor") == "fleur"
        tab.canvas.config(cursor="cross")
        app._set_mode("box")
        tab._on_alt_down()
        assert tab.canvas.cget("cursor") == "cross"

    def test_the_legend_help_and_badge_panels_are_rounded(self, app):
        tab = app._annotate_tab
        tab._legend_open = True
        app.show_help = True
        tab.render()
        canvas = tab.canvas
        assert all(canvas.type(i) == "polygon" for i in canvas.find_withtag("badge")
                   if canvas.type(i) != "text")
        legend_shapes = [canvas.type(i) for i in canvas.find_withtag("legend")
                         if canvas.type(i) not in ("text", "line", "oval")]
        assert legend_shapes and set(legend_shapes) == {"polygon"}
        assert not any(canvas.type(i) == "rectangle" and canvas.itemcget(i, "fill") == "#1A1A1A"
                       for i in canvas.find_all())

    def test_escape_does_not_deselect(self, app):
        tab, ann = polygon_setup(app)
        tab.select_annotation(ann.id)
        app._on_escape()
        assert app._selected_annotation_id == ann.id

    def test_escape_discards_a_paused_trace_whole(self, app):
        tab, _ = polygon_setup(app)
        app._stream_mode = True
        tab.on_button_press(click_at(tab, 300, 300))
        tab._on_motion(motion_at(tab, 310, 300))
        tab.on_button_press(click_at(tab, 320, 300))
        assert not app._stream_active and app.current_polygon
        app._on_escape()
        assert app.current_polygon == []

    def test_clicking_a_selected_polygons_vertex_starts_a_polygon_there(self, app):
        tab, ann = polygon_setup(app)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, 100, 100))
        tab.on_button_release(click_at(tab, 100, 100))
        assert app._selected_annotation_id is None
        assert app.current_polygon == [(100, 100)]
        assert app.document.get(ann.id).points == ann.points

    def test_dragging_a_selected_polygons_vertex_moves_it(self, app):
        tab, ann = polygon_setup(app)
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, 100, 100))
        tab.on_move_press(click_at(tab, 130, 120))
        tab.on_button_release(click_at(tab, 130, 120))
        assert app.document.get(ann.id).points[0] == (130, 120)
        assert app.current_polygon == []

    def test_snap_goes_to_vertices_never_to_edges(self, app):
        tab, _ = polygon_setup(app)
        app.snap_enabled = True
        assert tab._maybe_snap(150, 102) == (150, 102)
        assert tab._maybe_snap(103, 102) == (100, 100)

    def test_stream_spacing_is_measured_in_screen_pixels(self, app):
        tab, _ = polygon_setup(app, scale=4.0)
        app._stream_mode = True
        app.current_polygon = [(10, 10)]
        app._stream_active = True
        tab._on_motion(motion_at(tab, 11, 10))
        assert len(app.current_polygon) == 1
        tab._on_motion(motion_at(tab, 12, 10))
        assert app.current_polygon == [(10, 10), (12, 10)]

    def test_a_click_on_a_polygon_outline_selects_it_with_snap_on(self, app):
        tab, ann = polygon_setup(app)
        app.snap_enabled = True
        tab.on_button_press(click_at(tab, 150, 100))
        assert app._selected_annotation_id == ann.id
        assert app.current_polygon == []

    def test_a_click_that_snaps_to_a_vertex_starts_a_polygon_there(self, app):
        tab, ann = polygon_setup(app)
        app.snap_enabled = True
        tab.on_button_press(click_at(tab, 103, 102))
        assert app._selected_annotation_id is None
        assert app.current_polygon == [(100, 100)]

    def test_alt_click_selects_a_dense_polygon_with_snap_on(self, app):
        tab, _ = polygon_setup(app)
        app.snap_enabled = True
        dense = new_annotation(
            "polygon", tuple((x, 300) for x in range(300, 400, 5)) + ((350, 350),),
            app.active_class, "tester")
        app.document.add(dense)
        tab.on_button_press(click_at(tab, 322, 300))
        assert app.current_polygon == [(320, 300)]
        app._on_escape()
        tab.on_alt_press(click_at(tab, 322, 300))
        assert app._selected_annotation_id == dense.id
        assert app.current_polygon == []

    def test_alt_click_off_every_polygon_is_an_ordinary_press(self, app):
        tab, _ = polygon_setup(app)
        tab.on_alt_press(click_at(tab, 400, 400))
        assert app._selected_annotation_id is None
        assert app.current_polygon == [(400, 400)]

    def test_a_click_clear_of_every_vertex_and_outline_starts_a_polygon(self, app):
        tab, ann = polygon_setup(app)
        app.snap_enabled = True
        # 28 px from the (100, 100) vertex, outside both the snap and the outline radius.
        tab.on_button_press(click_at(tab, 80, 80))
        assert app._selected_annotation_id is None
        assert app.current_polygon == [(80, 80)]

    def test_streamed_vertices_snap_to_a_neighbour(self, app):
        tab, _ = polygon_setup(app)
        app.snap_enabled = True
        app._stream_mode = True
        app.current_polygon = [(80, 80)]
        app._stream_active = True
        tab._on_motion(motion_at(tab, 94, 94))
        assert app.current_polygon == [(80, 80), (100, 100)]

    def test_pausing_thins_the_streamed_run_but_keeps_its_anchors(self, app):
        tab, _ = polygon_setup(app)
        app._stream_mode = True
        tab.on_button_press(click_at(tab, 300, 300))
        for x in range(306, 400, 6):
            tab._on_motion(motion_at(tab, x, 300))
        assert len(app.current_polygon) > 10
        tab.on_button_press(click_at(tab, 396, 300))
        assert app.current_polygon == [(300, 300), (396, 300)]
        assert not app._stream_active

    def test_thinning_keeps_a_corner_and_the_vertices_before_the_run(self, app):
        tab, _ = polygon_setup(app)
        app._stream_mode = True
        app.current_polygon = [(200, 200), (250, 250)]
        tab.on_button_press(click_at(tab, 300, 300))
        for x in range(306, 400, 6):
            tab._on_motion(motion_at(tab, x, 300))
        for y in range(306, 400, 6):
            tab._on_motion(motion_at(tab, 396, y))
        tab._on_double_click(click_at(tab, 396, 396))
        closed = app.document.annotations[-1]
        assert closed.points == ((200, 200), (250, 250), (300, 300), (396, 300), (396, 396))

    def test_thinning_tolerance_is_measured_in_screen_pixels(self, app):
        tab, _ = polygon_setup(app, scale=4.0)
        app._stream_mode = True
        tab.on_button_press(click_at(tab, 300, 300))
        # A 3 image px bump is 12 screen px, inside the 15 px tolerance, so it goes.
        for x, y in ((302, 300), (304, 303), (306, 300), (308, 300)):
            tab._on_motion(motion_at(tab, x, y))
        tab.on_button_press(click_at(tab, 310, 300))
        assert app.current_polygon == [(300, 300), (308, 300)]

    def test_a_pause_click_while_streaming_is_snapped(self, app):
        tab, _ = polygon_setup(app)
        app.snap_enabled = True
        app._stream_mode = True
        app.current_polygon = [(80, 80), (90, 90)]
        app._stream_active = False
        tab.on_button_press(click_at(tab, 103, 102))
        assert app.current_polygon[-1] == (100, 100)
        assert app._stream_active

    def test_streaming_a_vertex_extends_the_drawing_without_a_full_render(self, app):
        tab, _ = polygon_setup(app)
        app._stream_mode = True
        tab.on_button_press(click_at(tab, 300, 300))
        renders = []
        tab.render = lambda: renders.append(True)
        before = len(tab.canvas.find_all())
        tab._on_motion(motion_at(tab, 310, 300))
        assert app.current_polygon == [(300, 300), (310, 300)]
        assert renders == []
        assert len(tab.canvas.find_all()) > before
        cx, cy = tab.image_to_canvas(310, 300)
        assert any(tab.canvas.coords(i)[-2:] == [cx, cy]
                   for i in tab.canvas.find_withtag("current_polygon")
                   if tab.canvas.type(i) == "line")

    def test_snap_ring_encloses_the_pointer_not_just_the_vertex(self, app):
        tab, _ = polygon_setup(app)
        app.snap_enabled = True
        tab._motion_last_time = 0
        tab._on_motion(motion_at(tab, 112, 100))
        assert tab._snap_indicator_item is not None
        x0, y0, x1, y1 = tab.canvas.coords(tab._snap_indicator_item)
        px, py = tab.image_to_canvas(112, 100)
        assert x0 < px < x1 and y0 < py < y1
        vx, vy = tab.image_to_canvas(100, 100)
        assert x0 < vx < x1 and y0 < vy < y1


class TestLegend:
    def test_legend_chip_opens_and_closes_without_drawing(self, app):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        tab.render()
        count = len(app.document.annotations)
        x0, y0, x1, y1 = tab._legend_bbox
        chip = type("E", (), {"x": (x0 + x1) / 2, "y": y1 - 2})()
        tab.on_button_press(chip)
        tab.on_button_release(chip)
        assert tab._legend_open
        tab.on_button_press(chip)
        tab.on_button_release(chip)
        assert not tab._legend_open
        assert len(app.document.annotations) == count and app.rect is None

    def legend_texts(self, tab):
        """Every text string the open legend panel draws."""
        canvas = tab.canvas
        return [canvas.itemcget(i, "text") for i in canvas.find_withtag("legend")
                if canvas.type(i) == "text"]

    def test_reviewing_keys_status_then_classes_then_marks(self, app):
        tab = app._annotate_tab
        tab._legend_open = True
        tab.render()
        texts = self.legend_texts(tab)
        assert "Review status: annotation | prediction" in texts
        assert "Accepted" in texts
        assert texts.index("Accepted") < texts.index("Class label") < texts.index("Marks")

    def test_hiding_predictions_keeps_the_review_status_section(self, app):
        tab = app._annotate_tab
        app._review_show_pred = False
        tab._legend_open = True
        tab.render()
        texts = self.legend_texts(tab)
        assert "Review status: annotation | prediction" in texts
        assert "Class label" in texts and "Marks" in texts

    def test_a_class_row_is_keyed_by_its_id_in_the_class_colour(self, app):
        tab = app._annotate_tab
        tab._legend_open = True
        tab.render()
        canvas = tab.canvas
        keyed = [i for i in canvas.find_withtag("legend")
                 if canvas.type(i) == "text" and canvas.itemcget(i, "text") == "0"]
        assert len(keyed) == 1
        assert canvas.itemcget(keyed[0], "fill") == app._get_class_color(0)
        assert "class_0" in self.legend_texts(tab)

    def test_the_class_rows_follow_what_the_canvas_draws(self, app):
        tab = app._annotate_tab
        tab._legend_open = True
        app._select_class_for_filter_and_draw(1)
        tab.render()
        texts = self.legend_texts(tab)
        assert "class_1" in texts and "class_0" not in texts
        app._on_class_selected("All")
        app._visible_var.set(False)
        app._on_visible_toggled()
        app._review_show_pred = False
        tab.render()
        assert "Class label" not in self.legend_texts(tab)

    def test_the_in_focus_row_survives_a_class_filter_that_empties_the_queue(self, app):
        tab = app._annotate_tab
        tab._legend_open = True
        app.class_names[7] = "empty_class"
        app._select_class_for_filter_and_draw(7)
        tab.render()
        assert app.queue == [] and "In focus" in self.legend_texts(tab)

    def test_an_image_with_no_predictions_drops_the_status_section_and_in_focus(self, app):
        tab = app._annotate_tab
        tab._legend_open = True
        tab.render()
        assert "In focus" in self.legend_texts(tab)
        app.go_to_image(1)
        tab._legend_open = True
        tab.render()
        texts = self.legend_texts(tab)
        assert app.shape_statuses is None
        assert "Review status: annotation | prediction" not in texts
        assert "In focus" not in texts and "Marks" in texts

    def test_the_snap_row_is_keyed_only_where_the_ring_can_appear(self, app):
        tab = app._annotate_tab
        tab._legend_open = True
        app.mode, app.snap_enabled = "polygon", True
        tab.render()
        assert "Snap target" in self.legend_texts(tab)
        app.snap_enabled = False
        tab.render()
        assert "Snap target" not in self.legend_texts(tab)
        app.mode, app.snap_enabled = "box", True
        tab.render()
        assert "Snap target" not in self.legend_texts(tab)


def in_widget(child, parent):
    """True when child sits anywhere under parent in the widget tree."""
    return str(child).startswith(f"{parent}.")


class TestBarLayout:
    def test_the_title_carries_the_name_zoom_time_and_user(self, app):
        app._apply_title()
        title = app.root.title()
        assert app.images[app.index] in title
        assert f"{int(app._annotate_tab.scale * 100)}%" in title
        assert app._image_elapsed in title and app._current_user in title

    def test_the_review_nav_group_sits_in_the_status_bar(self, app):
        panel = app._review_panel
        for w in (panel.reviewed_cb, panel.status_dd, panel.prev_item_btn,
                  panel.item_entry, panel.item_total_label, panel.next_item_btn):
            assert in_widget(w, app.status_bar)

    def test_blind_pass_moves_down_beside_the_prediction_toggle(self, app):
        assert in_widget(app.blind_cb, app.status_bar)
        assert in_widget(app._pred_cb, app.status_bar)

    def test_the_counts_move_off_the_nav_side(self, app):
        panel = app._review_panel
        assert in_widget(panel.counts_label, app.status_bar)
        assert not in_widget(panel.counts_label, panel.right)

    def test_the_title_follows_the_zoom(self, app):
        app._annotate_tab.scale = 4.0
        app._update_status()
        assert "400%" in app.root.title()

    def test_the_no_matches_title_survives_the_zoom_and_timer_paths(self, app):
        app._active_filter = "complete"
        app._rebuild_filter()
        assert app._filtered_indices == []
        app.update_title()
        app._update_status()
        app._update_timer_display()
        assert app.root.title() == "YoloLabeler - No matches"

    def test_loading_an_image_resets_the_elapsed_time_in_the_title(self, app):
        app._image_elapsed = "4:37"
        app.go_to_image(1)
        assert "4:37" not in app.root.title() and "0:00" in app.root.title()

    def test_opening_a_smaller_folder_from_a_high_index_does_not_crash(self, app, folder):
        one = folder / "one"
        one.mkdir()
        Image.new("RGB", (64, 48), "gray").save(one / "only.jpg")
        app.go_to_image(1)
        assert app.index == 1
        app._init_folder(str(one))
        app._annotate_tab.load_image()
        assert app.index == 0 and "only.jpg" in app.root.title()


class TestReviewedToggle:
    def focus_key(self, app, key):
        idx = next(i for i, q in enumerate(app.queue) if q.key == key)
        app._review_panel.focus_item(idx)

    def test_it_lights_when_the_focused_item_has_a_verdict(self, app):
        panel = app._review_panel
        tp = next(q for q in app.queue if q.kind == "tp")
        self.focus_key(app, tp.key)
        assert not panel.reviewed_var.get()
        app.accept_item()
        self.focus_key(app, tp.key)
        assert panel.reviewed_var.get()

    def test_unticking_clears_the_verdict_back_to_not_reviewed(self, app):
        panel = app._review_panel
        tp = next(q for q in app.queue if q.kind == "tp")
        self.focus_key(app, tp.key)
        app.accept_item()
        self.focus_key(app, tp.key)
        panel.reviewed_var.set(False)
        panel.on_reviewed_toggled()
        assert tp.key not in app.verdicts
        assert not panel.reviewed_var.get()

    def test_ticking_it_without_a_verdict_does_nothing(self, app):
        panel = app._review_panel
        tp = next(q for q in app.queue if q.kind == "tp")
        self.focus_key(app, tp.key)
        panel.reviewed_var.set(True)
        panel.on_reviewed_toggled()
        assert not panel.reviewed_var.get()
        assert tp.key not in app.verdicts


class TestRenderFocus:
    def test_both_shapes_of_the_focused_pair_are_haloed(self, app):
        tab = app._annotate_tab
        tp = next(q for q in app.queue if q.kind == "tp")
        app._review_panel.focus_item(app.queue.index(tp))
        tab.render()
        canvas = tab.canvas
        gt = canvas.find_withtag("gt_focus")
        pred = canvas.find_withtag("pred_focus")
        halo = canvas.find_withtag("focus_halo")
        assert len(gt) == 1 and len(pred) == 1 and len(halo) == 2
        assert canvas.itemcget(gt[0], "outline") == STATUS_COLORS["not_reviewed"]
        assert {canvas.itemcget(h, "outline") for h in halo} == {SELECTION_COLOR}
        assert {tuple(canvas.coords(h)) for h in halo} == {tuple(canvas.coords(gt[0])),
                                                           tuple(canvas.coords(pred[0]))}

    def test_accepting_turns_the_pair_green_and_hiding_predictions_keeps_the_status_colour(
            self, app):
        tab = app._annotate_tab
        panel = app._review_panel
        tp = next(q for q in app.queue if q.kind == "tp")
        panel.focus_item(app.queue.index(tp))
        app.accept_item()
        app._select_class_by_id(tp.annotation.class_id)
        tab.fit_to_window()
        tab.render()
        colors = [c for c, _ in shapes_at(tab, tp.annotation.points)]
        assert colors == [STATUS_COLORS["accepted"]] * 2
        app._review_show_pred = False
        tab.render()
        colors = [c for c, _ in shapes_at(tab, tp.annotation.points)]
        assert colors == [STATUS_COLORS["accepted"]]

    def test_an_image_with_no_predictions_keeps_class_colours(self, app):
        tab = app._annotate_tab
        app.shape_statuses = None
        ann = app.document.annotations[0]
        app._select_class_by_id(ann.class_id)
        tab.fit_to_window()
        tab.render()
        colors = [c for c, _ in shapes_at(tab, ann.points)]
        assert app._get_class_color(ann.class_id) in colors

    def test_editing_the_focused_pair_draws_it_as_selected(self, app):
        tab = app._annotate_tab
        tp = next(q for q in app.queue if q.kind == "tp")
        app._review_panel.focus_item(app.queue.index(tp))
        ann = app.queue[app.queue_index].annotation
        app.edit_pair()
        tab.fit_to_window()
        tab.render()
        canvas = tab.canvas
        assert canvas.find_withtag("gt_focus") == ()
        halo = canvas.find_withtag("focus_halo")
        pred = canvas.find_withtag("pred_focus")
        assert len(halo) == 1 and canvas.coords(halo[0]) == canvas.coords(pred[0])
        assert SELECTION_COLOR in [c for c, _ in shapes_at(tab, ann.points)]


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
        app.ask_incomplete_step = lambda: "continue"
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

    def test_load_image_keeps_the_active_class(self, app):
        tab, panel = app._annotate_tab, app._review_panel
        focused = app.queue[panel.first_unreviewed()]
        other = next(q.class_id for q in app.queue if q.class_id != focused.class_id)
        app._select_class_by_id(other)
        tab.load_image()
        assert app.active_class == other
        assert tab.scale != 1.0

    def test_manual_step_still_switches_the_active_class(self, app):
        panel = app._review_panel
        target = next(i for i, q in enumerate(app.queue) if q.class_id != app.active_class)
        panel.focus_item(target)
        assert app.active_class == app.queue[target].class_id


class TestItemStepper:
    def _type(self, panel, text):
        panel.item_entry.delete(0, "end")
        panel.item_entry.insert(0, text)
        panel._on_item_enter()

    def test_the_entry_shows_the_position_and_the_label_the_total(self, app):
        panel = app._review_panel
        panel.focus_item(1)
        assert panel.item_entry.get() == "2"
        assert panel.item_total_label.cget("text") == f"/ {len(app.queue)}"

    def test_typing_a_position_focuses_that_item(self, app):
        panel = app._review_panel
        panel.focus_item(1)
        self._type(panel, "1")
        assert app.queue_index == 0
        assert panel.item_entry.get() == "1"

    def test_an_out_of_range_or_unparsable_entry_reverts(self, app):
        panel = app._review_panel
        panel.focus_item(1)
        for text in (str(len(app.queue) + 1), "0", "-3", "x"):
            self._type(panel, text)
            assert app.queue_index == 1
            assert panel.item_entry.get() == "2"

    def test_an_empty_queue_leaves_the_entry_blank(self, app):
        panel = app._review_panel
        panel.type_var.set("FN")
        panel.on_type_changed("FN")
        assert app.queue == []
        assert panel.item_entry.get() == ""
        assert panel.item_total_label.cget("text") == "/ 0"


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


class TestClassRequiredToDraw:
    def test_box_draw_blocked_while_filter_is_all(self, app):
        tab = app._annotate_tab
        app._on_class_selected("All")
        tab.fit_to_window()
        before = len(app.document.annotations)
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert len(app.document.annotations) == before
        assert "class" in app.banner_text

    def test_polygon_draw_blocked_while_filter_is_all(self, poly_app):
        tab = poly_app._annotate_tab
        poly_app._on_class_selected("All")
        tab.fit_to_window()
        tab.on_button_press(click_at(tab, 500, 400))
        assert poly_app.current_polygon == []
        assert "class" in poly_app.banner_text

    def test_numeric_shortcut_sets_the_filter_and_unblocks_drawing(self, app):
        tab = app._annotate_tab
        app._on_class_selected("All")
        app.ACTIONS["class_0"]()
        assert app._review_filter_class == 0
        assert app.active_class == 0
        tab.fit_to_window()
        before = len(app.document.annotations)
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert len(app.document.annotations) == before + 1

    def test_existing_box_still_selectable_while_filter_is_all(self, app):
        tab, ann = box_setup(app)
        app._on_class_selected("All")
        tab.on_button_press(click_at(tab, *top_edge_mid(ann)))
        assert app._selected_annotation_id == ann.id


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
        assert panel.current_item().kind == "tp"
        assert panel.counts_label.cget("text") == "TP 1  FP 1  FN 0  (1 not reviewed)"
        tab.on_button_press(click_at(tab, vx, vy))
        tab.on_move_press(click_at(tab, 10, 10))
        tab.on_button_release(click_at(tab, 10, 10))
        # The drag alone pulls the GT below the IoU threshold, and the strip says so.
        # The verdict moves to the new FN, so both FPs are left, and focus follows the edit.
        assert panel.counts_label.cget("text") == "TP 0  FP 2  FN 1  (2 not reviewed)"
        assert panel.current_item().kind == "fn"
        assert key not in app.verdicts
        ann_id = panel.current_item().annotation.id
        assert {k: v for k, v in app.verdicts[ann_id].items() if k in ("action", "by", "at")} == {
            k: v for k, v in before.items() if k in ("action", "by", "at")}

    def test_drawing_an_unrelated_box_records_nothing_for_the_focused_item(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        panel.focus_item(len(app.queue) - 1)
        item = panel.current_item()
        assert item.kind == "tp"
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        assert len(app.document.annotations) == 2
        assert item.key not in app.verdicts
        assert panel.current_item().key == item.key

    def test_a_hand_drawn_box_is_accepted_without_a_review_action(self, app, folder):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        drawn = app.document.annotations[-1]
        assert app.verdicts[drawn.id]["action"] == "accepted"
        assert disk_verdicts(folder)[drawn.id]["action"] == "accepted"
        items = build_queue(app.document, app.predictions, app.matches, app.verdicts,
                            filter_status="not_reviewed")
        assert all(q.key != drawn.id for q in items)

    def test_a_hand_drawn_box_is_on_disk_before_its_verdict_is(self, app, folder):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        drawn = app.document.annotations[-1]
        lines = (folder / "labels" / "detect" / "a.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert disk_verdicts(folder)[drawn.id]["action"] == "accepted"

    def test_a_failed_save_records_no_verdict_and_rolls_the_accept_back(self, app, folder):
        import os
        broken = folder / "labels" / "detect" / "a.txt"
        os.remove(broken)
        os.makedirs(broken)
        try:
            app._review_panel.focus_item(0)
            assert app.queue[0].kind == "fp"
            app.accept_item()
            assert app.verdicts == {} and disk_verdicts(folder) == {}
            assert len(app.document.annotations) == 1
            assert app._redo_stack == []
            assert "Could not save" in app.banner_text
        finally:
            os.rmdir(broken)

    def test_undoing_a_hand_drawn_box_drops_its_verdict(self, app):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        drawn = app.document.annotations[-1]
        app.undo()
        assert drawn.id not in app.verdicts

    def test_rejecting_a_tp_does_not_select_the_deleted_annotation(self, app):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        item = panel.current_item()
        assert item.kind == "tp"
        ann_id = item.annotation.id
        app.reject_item()
        assert all(a.id != ann_id for a in app.document.annotations)
        assert app._selected_annotation_id != ann_id

    def test_accepting_an_fp_moves_on_without_selecting_anything(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        item = panel.current_item()
        assert item.kind == "fp"
        key = item.key
        app.accept_item()
        created = app.document.annotations[-1]
        assert created.source == "accepted"
        assert app._selected_annotation_id is None
        after = panel.current_item()
        assert after.key != key and after.key not in app.verdicts

    def test_edit_on_an_fp_accepts_it_and_selects_the_new_annotation(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        item = panel.current_item()
        assert item.kind == "fp"
        app.edit_pair()
        created = app.document.annotations[-1]
        assert created.source == "accepted" and created.prediction_id == item.key
        assert app.verdicts[item.key]["action"] == "accepted"
        assert app._selected_annotation_id == created.id
        assert panel.current_item().key == item.key

    def test_edit_turns_labels_back_on(self, app):
        panel = app._review_panel
        panel.focus_item(len(app.queue) - 1)
        app._visible_var.set(False)
        app._on_visible_toggled()
        app.edit_pair()
        assert app._annotation_visible and app._visible_var.get()
        assert app._selected_annotation_id == panel.current_item().annotation.id

    def test_accepted_fp_prediction_stays_drawn_after_moving_on(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        panel.focus_item(0)
        pred = panel.current_item().prediction
        app.accept_item()
        assert panel.current_item().key != pred.id
        # Undo the focus zoom, or the prediction is culled off-screen and the
        # assertion turns on canvas size rather than on what was drawn.
        tab.fit_to_window()
        tab.render()
        assert any(tab.canvas.coords(i) == pytest.approx(
                       [*tab.image_to_canvas(*pred.points[0]), *tab.image_to_canvas(*pred.points[1])])
                   for i in tab.canvas.find_withtag("pred") + tab.canvas.find_withtag("pred_focus"))


# ── right-click delete while an item is focused ─────────────────────────────

def box_center(ann):
    """The centre point of a box annotation."""
    (x1, y1), (x2, y2) = ann.points
    return (x1 + x2) / 2, (y1 + y2) / 2


def top_edge_mid(ann):
    """The midpoint of a box annotation's top edge, where a press picks the box."""
    (x1, y1), (x2, _) = ann.points
    return (x1 + x2) / 2, y1


class TestRightClickDelete:
    def test_deleting_the_focused_annotation_leaves_the_queue_usable(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.kind == "tp"))
        ann = panel.current_item().annotation
        assert ann is not None
        tab.on_right_click(click_at(tab, *top_edge_mid(ann)))
        assert all(a.id != ann.id for a in app.document.annotations)
        assert all(q.annotation is None or q.annotation.id != ann.id for q in app.queue)
        assert tab.canvas.find_withtag("gt_focus") == ()
        # Both of these read the focused item's annotation and used to raise KeyError.
        app.edit_pair()
        app.reject_item()

    def test_deleting_an_unrelated_annotation_keeps_the_focus(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.kind == "tp"))
        key = panel.current_item().key
        tab.fit_to_window()
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        drawn = app.document.annotations[-1]
        assert drawn.class_id == app.active_class
        tab.on_right_click(click_at(tab, *top_edge_mid(drawn)))
        assert all(a.id != drawn.id for a in app.document.annotations)
        assert panel.current_item().key == key
        assert panel.current_item().annotation is not None


# ── verdict carry-forward across a geometry edit ────────────────────────────

def focus_kind(app, kind):
    """Focus the first queue item of that kind and return it."""
    panel = app._review_panel
    panel.focus_item(next(i for i, q in enumerate(app.queue) if q.kind == kind))
    return panel.current_item()


def drag_box_to(tab, ann, ix, iy):
    """Move an existing box annotation by its outline so its centre lands near (ix, iy)."""
    tab.select_annotation(ann.id)
    ex, ey = top_edge_mid(ann)
    cx, cy = box_center(ann)
    tab.on_button_press(click_at(tab, ex, ey))
    tab.on_move_press(click_at(tab, ix + ex - cx, iy + ey - cy))
    tab.on_button_release(click_at(tab, ix + ex - cx, iy + ey - cy))


def unfiltered_item_for(app, ann_id):
    """The queue item holding that annotation, ignoring every active filter."""
    items = build_queue(app.document, app.predictions, app.matches, app.verdicts)
    return next(q for q in items if q.annotation is not None and q.annotation.id == ann_id)


class TestVerdictCarriesForward:
    def test_accepted_match_keeps_its_verdict_when_the_box_moves_away(self, app, folder):
        tab = app._annotate_tab
        pred_key = focus_kind(app, "tp").key
        app.accept_item()
        assert app.verdicts[pred_key]["action"] == "accepted"
        before_pred = dict(app.verdicts[pred_key])
        ann = focus_kind(app, "tp").annotation
        tab.fit_to_window()
        drag_box_to(tab, ann, 576, 432)
        moved = unfiltered_item_for(app, ann.id)
        assert moved.kind == "fn" and moved.key == ann.id
        assert app.verdicts[ann.id]["action"] == "accepted"
        assert app.verdicts[ann.id]["kind"] == "fn"
        assert app.verdicts[ann.id]["by"] == before_pred["by"]
        assert app.verdicts[ann.id]["at"] == before_pred["at"]
        # The prediction left behind is an unreviewed FP again, not a green one.
        assert pred_key not in app.verdicts
        assert disk_verdicts(folder)[ann.id]["action"] == "accepted"
        assert pred_key not in disk_verdicts(folder)
        assert app._review_panel.current_item().key == ann.id

    def test_an_edit_off_focus_still_moves_the_verdict(self, app):
        tab = app._annotate_tab
        pred_key = focus_kind(app, "tp").key
        app.accept_item()
        ann = unfiltered_item_for(app, next(
            a.id for a in app.document.annotations)).annotation
        focus_kind(app, "fp")
        assert app._review_panel.current_item().annotation is None
        tab.fit_to_window()
        drag_box_to(tab, ann, 576, 432)
        assert unfiltered_item_for(app, ann.id).kind == "fn"
        assert app.verdicts[ann.id]["action"] == "accepted"
        assert pred_key not in app.verdicts

    def test_deleting_a_polygon_vertex_moves_the_verdict(self, poly_app):
        app = poly_app
        tab = app._annotate_tab
        ann = app.document.annotations[0]
        # Shifted right so the match holds at IoU 0.52 and drops to 0.19 once a corner goes.
        app._engine.set_points(ann.id, ((232, 144), (360, 144), (360, 240), (232, 240)))
        app._review_panel.refresh(keep_focus=False)
        pred_key = focus_kind(app, "tp").key
        app.accept_item()
        tab.fit_to_window()
        tab.select_annotation(ann.id)
        tab.on_right_click(click_at(tab, 232, 144))
        assert len(app.document.get(ann.id).points) == 3
        assert unfiltered_item_for(app, ann.id).kind == "fn"
        assert app.verdicts[ann.id]["action"] == "accepted"
        assert pred_key not in app.verdicts

    def test_deleting_a_matched_annotation_drops_its_verdict(self, app):
        tab = app._annotate_tab
        item = focus_kind(app, "tp")
        app.accept_item()
        assert item.key in app.verdicts
        app._select_class_for_filter_and_draw(0)
        tab.fit_to_window()
        tab.on_right_click(click_at(tab, *top_edge_mid(item.annotation)))
        assert all(a.id != item.annotation.id for a in app.document.annotations)
        assert item.key not in app.verdicts

    def test_rejecting_the_selected_annotation_clears_the_selection(self, app):
        item = focus_kind(app, "tp")
        app._annotate_tab.select_annotation(item.annotation.id)
        app.reject_item()
        assert app._selected_annotation_id is None

    def test_an_edit_that_stays_matched_adds_no_verdict(self, app):
        tab = app._annotate_tab
        pred_key = focus_kind(app, "tp").key
        app.accept_item()
        before = {k: dict(v) for k, v in app.verdicts.items()}
        ann = focus_kind(app, "tp").annotation
        tab.fit_to_window()
        cx, cy = box_center(ann)
        drag_box_to(tab, ann, cx + 4, cy + 3)
        assert unfiltered_item_for(app, ann.id).kind == "tp"
        assert app.verdicts == before
        assert app._review_panel.current_item().key == pred_key

    def test_an_unreviewed_edit_invents_no_verdict(self, app):
        tab = app._annotate_tab
        item = focus_kind(app, "tp")
        ann = item.annotation
        assert item.key not in app.verdicts
        tab.fit_to_window()
        drag_box_to(tab, ann, 576, 432)
        assert unfiltered_item_for(app, ann.id).kind == "fn"
        assert ann.id not in app.verdicts


# ── overview zoom when a sweep completes ────────────────────────────────────

def at_overview(app):
    tab = app._annotate_tab
    cw, ch = tab.canvas.winfo_width(), tab.canvas.winfo_height()
    return (tab.scale == guimod.OVERVIEW_ZOOM
            and tab.offset_x == pytest.approx((cw - app.img_width * tab.scale) / 2)
            and tab.offset_y == pytest.approx((ch - app.img_height * tab.scale) / 2))


class TestSweepCompleteOverview:
    def test_reviewed_filter_does_not_zoom_out_on_a_re_accept(self, app):
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
        assert not at_overview(app)

    def test_last_unreviewed_item_zooms_out_and_stays_on_the_image(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        assert panel.current_item().kind == "fp"
        app.accept_item()
        panel.on_status_changed("Not reviewed")
        assert [q.kind for q in app.queue] == ["tp"]
        panel.focus_item(0)
        app.accept_item()
        assert app.images[app.index] == "a.jpg"
        assert at_overview(app)

    def test_type_filter_zooms_out_when_that_type_is_done(self, app, folder):
        panel = app._review_panel
        panel.on_type_changed("FP")
        assert [q.kind for q in app.queue] == ["fp"]
        fp_key = app.queue[0].key
        panel.focus_item(0)
        app.reject_item()
        assert app.images[app.index] == "a.jpg"
        assert at_overview(app)
        assert list(disk_verdicts(folder)) == [fp_key]

    def test_class_filter_zooms_out_when_that_class_is_done(self, app, folder):
        panel = app._review_panel
        app._review_filter_class = 1
        panel.refresh(keep_focus=False)
        assert [q.kind for q in app.queue] == ["fp"]
        fp_key = app.queue[0].key
        panel.focus_item(0)
        app.reject_item()
        assert app.images[app.index] == "a.jpg"
        assert at_overview(app)
        assert list(disk_verdicts(folder)) == [fp_key]

    def test_an_unfinished_sweep_focuses_the_next_item_instead(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        app.accept_item()
        assert not at_overview(app)
        assert panel.current_item().key not in app.verdicts


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
        app._select_class_for_filter_and_draw(0)
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

    def test_title_shows_current_image_name(self, app):
        assert app.images[app.index] in app.root.title()
        assert app.go_to_image(1)
        assert app.images[app.index] in app.root.title()

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

    def test_an_empty_folder_leaves_no_stale_queue_behind(self, app, folder, monkeypatch):
        app._review_panel.focus_item(0)
        empty = folder / "empty"
        empty.mkdir()
        monkeypatch.setattr(guimod.filedialog, "askdirectory", lambda **kw: str(empty))
        app._open_folder()
        assert app.document is None and app.queue == [] and app.predictions == []
        # Each of these used to index images[0] of an empty list.
        app._review_panel.on_type_changed("FP")
        app._review_panel.on_status_changed("Reviewed")
        app.accept_item()
        app.reject_item()
        app.edit_pair()
        assert app._review_panel.accept_btn.cget("state") == "disabled"

    def test_a_corrupt_sidecar_is_reported_and_the_image_stays_editable(self, app, folder):
        side = folder / "state" / "annotations" / "a.json"
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text("{not json", encoding="utf-8")
        app._annotate_tab.load_image()
        assert "sidecar could not be read" in app.banner_text
        assert app.load_errors == [] and len(app.document.annotations) == 1
        assert app._editable()


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

        def after(self, *a, **k):
            pass

    monkeypatch.setattr(guimod, "FitToContentInputDialog", Dialog)


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
        assert app.complete_cb.cget("text") == "Completed"
        assert "2 not reviewed" in app._review_panel.counts_label.cget("text")
        app._review_panel.focus_item(0)
        app.reject_item()
        assert app.complete_cb.cget("text") == "Completed"
        assert "1 not reviewed" in app._review_panel.counts_label.cget("text")

    def test_blind_hides_predictions_until_complete(self, app):
        app._blind_var.set(True)
        app._on_blind_toggled()
        assert app.predictions == [] and app.predictions_blind and app.queue == []
        app._complete_var.set(True)
        app._on_complete_toggled()
        assert app._stats_store.completion("a.jpg")["blind"] is True
        assert len(app.predictions) == 2 and not app.predictions_blind

    def test_blind_greys_out_inert_controls(self, app):
        panel = app._review_panel
        app._blind_var.set(True)
        app._on_blind_toggled()
        assert panel.type_dd.cget("state") == "disabled"
        assert panel.status_dd.cget("state") == "disabled"
        assert panel.conf_entry.cget("state") == "disabled"
        assert panel.item_entry.cget("state") == "disabled"
        assert panel.item_total_label.cget("text") == "/ 0"
        assert panel.prev_item_btn.cget("state") == "disabled"
        assert panel.next_item_btn.cget("state") == "disabled"
        app._blind_var.set(False)
        app._on_blind_toggled()
        assert panel.type_dd.cget("state") == "readonly"
        assert panel.status_dd.cget("state") == "readonly"
        assert panel.conf_entry.cget("state") == "normal"
        assert panel.item_entry.cget("state") == "normal"
        assert panel.prev_item_btn.cget("state") == "normal"
        assert panel.next_item_btn.cget("state") == "normal"
        assert panel.conf_entry.get() == f"{app.conf_threshold:.2f}"

    def test_blind_clears_the_prediction_layer_at_once(self, app):
        canvas = app.canvas
        assert canvas.find_withtag("pred") or canvas.find_withtag("pred_focus")
        app._blind_var.set(True)
        app._on_blind_toggled()
        # No further redraw: hiding the model's output is the whole point of the toggle.
        assert not canvas.find_withtag("pred")
        assert not canvas.find_withtag("pred_focus")

    def test_blind_does_not_reset_a_completed_review_status(self, app, folder):
        panel = app._review_panel
        panel.focus_item(0)
        app.accept_item()
        panel.focus_item(panel.first_unreviewed())
        app.accept_item()
        assert app.images[app.index] == "a.jpg"
        assert disk_img_status(folder) == "completed"
        assert app.go_to_image(0)
        app._blind_var.set(True)
        app._on_blind_toggled()
        assert app.predictions_blind and app.queue == []
        assert disk_img_status(folder) == "completed"

    def test_model_name_from_manifest(self, app, folder):
        from yololabeler.predictions.store import write_manifest
        write_manifest(folder / "predictions", {"model": "nathan_v15"})
        app._complete_var.set(True)
        app._on_complete_toggled()
        assert app._stats_store.completion("a.jpg")["model"] == "nathan_v15"


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

    def test_a_read_only_image_refuses_a_drawn_box(self, app, folder):
        p = folder / "labels" / "detect" / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\nnope\n", encoding="utf-8")
        app._annotate_tab.load_image()
        app.clear_banner()
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        assert len(app.document.annotations) == 1
        assert app.banner_text.startswith("Read-only")

    def test_a_read_only_image_refuses_a_verdict(self, app, folder):
        p = folder / "labels" / "detect" / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\nnope\n", encoding="utf-8")
        app._annotate_tab.load_image()
        app._review_panel.focus_item(0)
        assert app.queue[0].kind == "fp"
        app.accept_item()
        assert app.verdicts == {} and len(app.document.annotations) == 1
        assert app.banner_text.startswith("Read-only")

    def test_a_corrupt_manifest_is_quarantined_and_reported(self, app, folder):
        (folder / "predictions" / "manifest.json").write_text("{not json", encoding="utf-8")
        app._init_folder(str(folder))
        assert not (folder / "predictions" / "manifest.json").exists()
        assert list((folder / "predictions").glob("manifest.json.corrupt-*"))
        assert "manifest.json could not be read" in app.banner_text

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
        app._select_class_for_filter_and_draw(0)
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


# ── confirm before stepping past an incomplete image ────────────────────────

class TestIncompleteStep:
    def test_mark_complete_and_continue(self, app):
        app.ask_incomplete_step = lambda: "complete"
        app._annotate_tab.next_image()
        assert app.images[app.index] == "b.jpg"
        assert app._stats_store.completion("a.jpg") is not None
        assert "a.jpg" in app._completed_images

    def test_continue_without_marking(self, app):
        app.ask_incomplete_step = lambda: "continue"
        app._annotate_tab.next_image()
        assert app.images[app.index] == "b.jpg"
        assert app._stats_store.completion("a.jpg") is None

    def test_stay(self, app):
        app.ask_incomplete_step = lambda: "stay"
        app._annotate_tab.next_image()
        assert app.images[app.index] == "a.jpg"
        assert app._stats_store.completion("a.jpg") is None

    def test_a_complete_image_steps_without_asking(self, app):
        app._complete_var.set(True)
        app._on_complete_toggled()
        asked = []
        app.ask_incomplete_step = lambda: asked.append(True) or "stay"
        app._annotate_tab.next_image()
        assert asked == []
        assert app.images[app.index] == "b.jpg"

    def test_previous_image_never_asks(self, app):
        app.ask_incomplete_step = lambda: "continue"
        app._annotate_tab.next_image()
        asked = []
        app.ask_incomplete_step = lambda: asked.append(True) or "stay"
        app._annotate_tab.prev_image()
        assert asked == []
        assert app.images[app.index] == "a.jpg"

    def press_in_dialog(self, app, sequence):
        """Arm a key press on the not-complete dialog once it is open, then open it.

        The press waits out the title-bar callbacks CTkToplevel arms in its first
        200 ms, which call update() on the window; destroying it under one of
        those is an access violation, not a Python error.
        """
        def send():
            for w in app.root.winfo_children():
                if isinstance(w, tk.Toplevel) and w.title() == "Image not marked completed":
                    w.event_generate(sequence)
                    return
            app.root.after(50, send)
        app.root.after(400, send)
        return app._ask_incomplete_step()

    def test_right_in_the_dialog_continues_without_marking(self, app):
        assert self.press_in_dialog(app, "<Right>") == "continue"

    def test_enter_in_the_dialog_marks_completed(self, app):
        assert self.press_in_dialog(app, "<Return>") == "complete"

    def test_escape_in_the_dialog_stays(self, app):
        assert self.press_in_dialog(app, "<Escape>") == "stay"


# ── dialogs across a monitor DPI change ─────────────────────────────────────

def dialog_with_wide_text(cls, root):
    dialog = cls(root)
    dialog.resizable(False, False)
    ctk.CTkLabel(dialog, text="Mark this image complete and move to the next image?",
                 font=("Arial", 12)).pack(padx=16, pady=(16, 12))
    for text in ("Mark completed and continue (Enter)", "Continue without marking",
                 "Stay (Esc)"):
        ctk.CTkButton(dialog, text=text, width=260, font=("Arial", 12)).pack(
            padx=16, pady=(0, 8))
    dialog.update()
    return dialog


def rescale(dialog, factor):
    """Apply CustomTkinter's per-window DPI rescale, then its delayed min/max release."""
    from customtkinter.windows.widgets.scaling.scaling_tracker import ScalingTracker
    ScalingTracker.window_dpi_scaling_dict[dialog] *= factor
    ScalingTracker.update_scaling_callbacks_for_window(dialog)
    dialog._set_scaled_min_max()
    dialog.update()


class TestFitToContentDialog:
    def test_dialog_still_fits_its_contents_after_moving_monitors(self, app):
        dialog = dialog_with_wide_text(guimod.FitToContentToplevel, app.root)
        try:
            for factor in (1.5, 1 / 1.5, 1.25):
                rescale(dialog, factor)
                assert dialog.winfo_width() >= dialog.winfo_reqwidth()
                assert dialog.winfo_height() >= dialog.winfo_reqheight()
        finally:
            dialog.destroy()


# ── moving on along the path ────────────────────────────────────────────────

class TestNextUnreviewedAfter:
    def test_continues_forward_from_the_judged_item(self, app):
        panel = app._review_panel
        path = [q.key for q in app.queue]
        assert panel.next_unreviewed_after(path, path[0]) == 1

    def test_wraps_to_the_start(self, app):
        panel = app._review_panel
        path = [q.key for q in app.queue]
        assert panel.next_unreviewed_after(path, path[-1]) == 0

    def test_skips_reviewed_items(self, app):
        panel = app._review_panel
        path = [q.key for q in app.queue]
        app.verdicts[path[1]] = {"action": "accepted"}
        assert panel.next_unreviewed_after(path, path[1]) == 0


# ── comment and flag ────────────────────────────────────────────────────────

class TestCommentFlag:
    def test_saving_a_comment_flags_the_focused_item(self, app, folder):
        panel = app._review_panel
        panel.focus_item(0)
        key = panel.current_item().key
        app.ask_flag_comment = lambda heading, existing: ("save", "leaf or bur?")
        app.comment_on_item()
        assert set(app.flag_markers) == {key}
        assert "1 flagged" in panel.counts_label.cget("text")
        data = json.loads((folder / "state" / "review_stats.json").read_text(encoding="utf-8"))
        (entry,) = data["image"]["a.jpg"]["flags"][key]
        assert entry["comment"] == "leaf or bur?" and not entry["resolved"]
        assert key not in app.verdicts

    def test_flag_marker_and_badge_are_drawn(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        panel.focus_item(0)
        assert panel.current_item().kind == "fp"
        app.ask_flag_comment = lambda heading, existing: ("save", "")
        app.comment_on_item()
        tab.render()
        labels = [tab.canvas.itemcget(i, "text") for i in tab.canvas.find_all()
                  if tab.canvas.type(i) == "text"]
        # The focused FP's own label ends with the mark; nothing floats beside it.
        assert any(t.endswith(" ?") and "(0.80)" in t for t in labels)
        assert not tab.canvas.find_withtag("flag")
        badge = [tab.canvas.itemcget(i, "text") for i in tab.canvas.find_withtag("badge")
                 if tab.canvas.type(i) == "text"]
        assert badge[0].endswith("flagged")
        # Once focus moves on, the FP keeps a label of its own, mark included.
        panel.step(1)
        tab.render()
        fp_labels = [tab.canvas.itemcget(i, "text") for i in tab.canvas.find_withtag("pred_label")]
        assert fp_labels and all(t == "1: class_1 (0.80) ?" for t in fp_labels)

    def test_a_flagged_annotation_label_ends_with_the_mark(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.kind == "tp"))
        app.ask_flag_comment = lambda heading, existing: ("save", "")
        app.comment_on_item()
        app._select_class_for_filter_and_draw(0)
        app.queue = []
        tab.render()
        labels = [tab.canvas.itemcget(i, "text") for i in tab.canvas.find_all()
                  if tab.canvas.type(i) == "text"]
        assert f"0: {app.class_names[0]} ?" in labels

    def test_c_comments_on_the_selected_annotation_over_the_focused_item(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        app._select_class_for_filter_and_draw(1)
        panel.focus_item(0)
        assert panel.current_item().kind == "fp"
        focused_key = panel.current_item().key
        tab.fit_to_window()
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        drawn = app.document.annotations[-1]
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.key == focused_key))
        tab.select_annotation(drawn.id)
        headings = []
        app.ask_flag_comment = lambda heading, existing: headings.append(heading) or ("save", "?")
        app.comment_on_item()
        assert headings[0].startswith("FN")
        assert app._review.open_flag("a.jpg", drawn.id)["kind"] == "fn"
        assert app._review.open_flag("a.jpg", focused_key) is None

    def test_reopening_passes_the_open_flag_and_resolve_clears_it(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        app.ask_flag_comment = lambda heading, existing: ("save", "first")
        app.comment_on_item()
        seen = []
        app.ask_flag_comment = lambda heading, existing: seen.append(existing) or ("resolve", None)
        app.comment_on_item()
        assert seen[0]["comment"] == "first"
        assert app.flag_markers == {}

    def test_cancel_changes_nothing(self, app):
        app._review_panel.focus_item(0)
        app.ask_flag_comment = lambda heading, existing: ("cancel", None)
        app.comment_on_item()
        assert app.flag_markers == {}
        assert app._review.flags("a.jpg") == {}

    def test_flagged_review_status_filter(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        key = panel.current_item().key
        app.ask_flag_comment = lambda heading, existing: ("save", "")
        app.comment_on_item()
        panel.on_status_changed("Flagged")
        assert [q.key for q in app.queue] == [key]

    def test_completion_counts_a_flag_on_a_prediction_hidden_by_the_threshold(self, app):
        panel = app._review_panel
        panel.focus_item(0)
        item = panel.current_item()
        assert item.kind == "fp" and item.prediction.confidence == pytest.approx(0.8)
        app.ask_flag_comment = lambda heading, existing: ("save", "")
        app.comment_on_item()
        panel.set_threshold(0.85)
        assert app.flag_markers == {}
        app._complete_var.set(True)
        app._on_complete_toggled()
        assert app._stats_store.completion("a.jpg")["open_flags"] == 1

    def test_completion_records_open_flags_and_image_filter_finds_the_image(self, app):
        app._review_panel.focus_item(0)
        app.ask_flag_comment = lambda heading, existing: ("save", "")
        app.comment_on_item()
        app._complete_var.set(True)
        app._on_complete_toggled()
        assert app._stats_store.completion("a.jpg")["open_flags"] == 1
        app._on_filter_changed("Flagged")
        assert app._filtered_indices == [0]



class TestFlagResolvedByRemoval:
    def draw_unflagged_fn(self, app):
        tab = app._annotate_tab
        app._select_class_for_filter_and_draw(0)
        tab.fit_to_window()
        tab.on_button_press(click_at(tab, 420, 360))
        tab.on_button_release(click_at(tab, 520, 440))
        drawn = app.document.annotations[-1]
        app.verdicts.pop(drawn.id, None)
        app._review_panel.refresh(keep_focus=False)
        return drawn

    def flag_focused(self, app, key):
        panel = app._review_panel
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.key == key))
        app.ask_flag_comment = lambda heading, existing: ("save", "unsure")
        app.comment_on_item()
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.key == key))

    def test_rejecting_a_flagged_fn_resolves_its_flag(self, app):
        drawn = self.draw_unflagged_fn(app)
        self.flag_focused(app, drawn.id)
        app.reject_item()
        (entry,) = app._review.flags("a.jpg")[drawn.id]
        assert entry["resolved"] and entry["resolved_note"] == "rejected"
        assert not app._review.has_open_flags("a.jpg")
        assert app.flag_markers == {}

    def test_rejecting_a_flagged_tp_keeps_the_flag_open_on_its_prediction(self, app):
        key = next(q.key for q in app.queue if q.kind == "tp")
        self.flag_focused(app, key)
        app.reject_item()
        assert app._review.open_flag("a.jpg", key) is not None
        assert key in app.flag_markers

    def test_deleting_a_flagged_box_resolves_its_flag(self, app):
        drawn = self.draw_unflagged_fn(app)
        self.flag_focused(app, drawn.id)
        tab = app._annotate_tab
        tab.on_right_click(click_at(tab, *top_edge_mid(drawn)))
        (entry,) = app._review.flags("a.jpg")[drawn.id]
        assert entry["resolved"] and entry["resolved_note"] == "deleted"

    def test_resolving_the_last_flag_drops_the_image_from_the_flagged_list(self, app):
        drawn = self.draw_unflagged_fn(app)
        self.flag_focused(app, drawn.id)
        app._on_filter_changed("Flagged")
        assert app._filtered_indices == [0]
        self.flag_focused(app, drawn.id)
        app.reject_item()
        assert app._filtered_indices == []


class TestFlagWithoutQueue:
    def test_c_flags_the_selected_annotation_on_a_blind_image(self, app):
        app._blind_var.set(True)
        app._on_blind_toggled()
        assert app.queue == []
        ann = app.document.annotations[0]
        app._annotate_tab.select_annotation(ann.id)
        headings = []
        app.ask_flag_comment = lambda heading, existing: headings.append(heading) or ("save", "")
        app.comment_on_item()
        assert headings[0].startswith("Annotation")
        assert app._review.open_flag("a.jpg", ann.id)["kind"] is None
        assert set(app.flag_markers) == {ann.id}

    def test_that_flag_shows_on_the_tp_once_predictions_are_back(self, app):
        app._blind_var.set(True)
        app._on_blind_toggled()
        ann = app.document.annotations[0]
        app._annotate_tab.select_annotation(ann.id)
        app.ask_flag_comment = lambda heading, existing: ("save", "")
        app.comment_on_item()
        app._blind_var.set(False)
        app._on_blind_toggled()
        tp = next(q for q in app.queue if q.kind == "tp")
        assert tp.annotation.id == ann.id and ann.id in app.flag_markers
        app._review_panel.focus_item(app.queue.index(tp))
        seen = []
        app.ask_flag_comment = lambda heading, existing: seen.append(existing) or ("resolve", None)
        app.comment_on_item()
        assert seen[0] is not None and app.flag_markers == {}

    def test_nothing_selected_and_no_queue_shows_a_banner(self, app):
        app.ask_incomplete_step = lambda: "continue"
        app._annotate_tab.next_image()
        assert app.queue == []
        app.comment_on_item()
        assert app.banner_text == "Select an annotation or focus a review item to comment on it."
