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
from yololabeler.review.layer import SELECTION_COLOR  # noqa: E402


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
        app._select_class_by_id(0)
        app.class_names[1] = "other"
        app._select_class_by_id(1)
        assert app._annotate_tab.visible_annotations() == []


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

    def test_dragging_the_body_shifts_both_points_by_the_same_delta(self, app):
        tab, ann = box_setup(app)
        (x1, y1), (x2, y2) = ann.points
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
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

    def test_dragging_the_body_past_an_edge_clamps_it_inside_the_image(self, app):
        tab, ann = box_setup(app)
        (x1, y1), (x2, y2) = ann.points
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tab.select_annotation(ann.id)
        tab.on_button_press(click_at(tab, cx, cy))
        tab.on_move_press(click_at(tab, x2 + 400, y2 + 400))
        tab.on_button_release(click_at(tab, x2 + 400, y2 + 400))
        assert flat(app.document.get(ann.id).points) == pytest.approx(
            [app.img_width - (x2 - x1), app.img_height - (y2 - y1),
             app.img_width, app.img_height])

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
        (x1, y1), (x2, y2) = ann.points
        tab.on_button_press(click_at(tab, (x1 + x2) / 2, (y1 + y2) / 2))
        assert app._selected_annotation_id == ann.id
        assert app.rect is None and app._box_edit_mode is None
        tab.on_button_release(click_at(tab, (x1 + x2) / 2, (y1 + y2) / 2))
        assert len(app.document.annotations) == count

    def test_clicking_inside_a_box_with_nothing_selected_selects_it(self, app):
        tab, ann = box_setup(app)
        assert app._selected_annotation_id is None and app.mode == "box"
        count = len(app.document.annotations)
        (x1, y1), (x2, y2) = ann.points
        tab.on_button_press(click_at(tab, (x1 + x2) / 2, (y1 + y2) / 2))
        tab.on_button_release(click_at(tab, (x1 + x2) / 2, (y1 + y2) / 2))
        assert app._selected_annotation_id == ann.id
        assert app.rect is None
        assert len(app.document.annotations) == count

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


class TestRenderFocus:
    def test_focused_annotation_is_haloed_and_keeps_its_class_colour(self, app):
        tab = app._annotate_tab
        app._review_panel.focus_item(len(app.queue) - 1)
        ann = app.queue[app.queue_index].annotation
        assert ann is not None
        tab.render()
        canvas = tab.canvas
        gt = canvas.find_withtag("gt_focus")
        halo = canvas.find_withtag("focus_halo")
        assert len(gt) == 1 and len(halo) == 1
        assert canvas.itemcget(gt[0], "outline") == app._get_class_color(ann.class_id)
        assert canvas.itemcget(halo[0], "outline") == SELECTION_COLOR
        assert canvas.coords(halo[0]) == canvas.coords(gt[0])

    def test_editing_the_focused_pair_draws_it_as_selected(self, app):
        tab = app._annotate_tab
        app._review_panel.focus_item(len(app.queue) - 1)
        ann = app.queue[app.queue_index].annotation
        app.edit_pair()
        tab.render()
        assert tab.canvas.find_withtag("gt_focus") == ()
        assert tab.canvas.find_withtag("focus_halo") == ()
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
        cx, cy = box_center(ann)
        tab.on_button_press(click_at(tab, cx, cy))
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
        assert panel.item_label.cget("text").startswith("TP")
        assert panel.counts_label.cget("text") == "TP 1  FP 1  FN 0  (1 not reviewed)"
        tab.on_button_press(click_at(tab, vx, vy))
        tab.on_move_press(click_at(tab, 10, 10))
        tab.on_button_release(click_at(tab, 10, 10))
        # The drag alone pulls the GT below the IoU threshold, and the strip says so.
        assert panel.counts_label.cget("text") == "TP 0  FP 2  FN 1  (2 not reviewed)"
        assert panel.item_label.cget("text").startswith("FP")
        assert app.verdicts[key] == before

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
        tab.render()
        assert any(tab.canvas.coords(i) == pytest.approx(
                       [*tab.image_to_canvas(*pred.points[0]), *tab.image_to_canvas(*pred.points[1])])
                   for i in tab.canvas.find_withtag("pred") + tab.canvas.find_withtag("pred_focus"))


# ── right-click delete while an item is focused ─────────────────────────────

def box_center(ann):
    """The centre point of a box annotation."""
    (x1, y1), (x2, y2) = ann.points
    return (x1 + x2) / 2, (y1 + y2) / 2


class TestRightClickDelete:
    def test_deleting_the_focused_annotation_leaves_the_queue_usable(self, app):
        panel, tab = app._review_panel, app._annotate_tab
        panel.focus_item(next(i for i, q in enumerate(app.queue) if q.kind == "tp"))
        ann = panel.current_item().annotation
        assert ann is not None
        tab.on_right_click(click_at(tab, *box_center(ann)))
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
        tab.on_right_click(click_at(tab, *box_center(drawn)))
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
    """Move an existing box annotation by dragging its centre to (ix, iy)."""
    tab.select_annotation(ann.id)
    tab.on_button_press(click_at(tab, *box_center(ann)))
    tab.on_move_press(click_at(tab, ix, iy))
    tab.on_button_release(click_at(tab, ix, iy))


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
        assert app.verdicts[pred_key] == before_pred
        assert disk_verdicts(folder)[ann.id]["action"] == "accepted"

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

    def test_auto_advance_skips_an_image_the_list_filter_hides(self, app, folder):
        Image.new("RGB", (640, 480), "gray").save(folder / "c.jpg")
        app._init_folder(str(folder))
        app._annotate_tab.load_image()
        for name, status in (("a.jpg", "partial"), ("b.jpg", "complete"), ("c.jpg", "partial")):
            app._stats_store.set_image_status(name, status)
        app.filter_var.set("Partial")
        app._on_filter_changed("Partial")
        assert app._filtered_indices == [0, 2] and app.images[app.index] == "a.jpg"
        panel = app._review_panel
        panel.focus_item(0)
        app.accept_item()
        panel.focus_item(panel.first_unreviewed())
        app.accept_item()
        assert app.index in app._filtered_indices
        assert app.images[app.index] == "c.jpg"

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
        assert app.complete_cb.cget("text") == "Complete"
        assert "2 not reviewed" in app._review_panel.counts_label.cget("text")
        app._review_panel.focus_item(0)
        app.reject_item()
        assert app.complete_cb.cget("text") == "Complete"
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
        assert panel.prev_item_btn.cget("state") == "disabled"
        assert panel.next_item_btn.cget("state") == "disabled"
        app._blind_var.set(False)
        app._on_blind_toggled()
        assert panel.type_dd.cget("state") == "readonly"
        assert panel.status_dd.cget("state") == "readonly"
        assert panel.conf_entry.cget("state") == "normal"
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
        assert app.images[app.index] == "b.jpg"
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
