"""AnnotateTab — Annotation canvas, interaction, rendering for the Annotate tab.

Consolidates all annotate-tab functionality: canvas construction, bindings,
coordinate conversion, pan/zoom, image loading, box/polygon interaction,
snapping, vertex streaming, undo/redo, save, and canvas rendering
(absorbed from AnnotateRenderer).
"""

import math
import os
import sys
import time
import tkinter as tk
import tkinter.font as tkFont

from PIL import Image, ImageTk

from yololabeler.label_io import (
    parse_detect_labels, parse_segment_labels,
)
from yololabeler.matching import point_to_segment_dist, point_in_polygon
from yololabeler.rendering import halo_text
from yololabeler.utils import auto_orient_image

# Constants
VERTEX_HANDLE_RADIUS = 4
STREAM_MIN_DISTANCE = 6
SNAP_RADIUS = 15
FG_COLOR = "#E0E0E0"
CANVAS_BG = "#2D2D2D"


class AnnotateTab:
    """Annotate canvas: drawing, zoom/pan, snapping, rendering, undo/redo."""

    def __init__(self, app):
        self.app = app
        self.engine = app._engine
        self.canvas: tk.Canvas = None  # type: ignore[assignment]  # set in build()

        # View transform
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._cached_scale = None
        self._cached_tk_image = None

        self.zoom_levels = [
            0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67,
            0.75, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0,
        ]
        self.zoom_index = 0

        # Middle-click pan
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_start_offset_x = None
        self.pan_start_offset_y = None

        # Display throttling
        self._redraw_pending = False
        self._motion_last_time = 0.0
        self._resize_after_id = None
        self._fast_resample = False

        # Mouse tracking
        self._mouse_canvas_x = 0
        self._mouse_canvas_y = 0
        self._poly_preview_line = None
        self._snap_indicator_item = None

    def build(self, parent):
        """Create the annotate canvas and bind events."""
        self.canvas = tk.Canvas(
            parent, cursor="cross", bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._setup_bindings()

    def _setup_bindings(self):
        """Bind canvas-level mouse/scroll events."""
        assert self.canvas is not None
        c = self.canvas
        c.bind("<Configure>", self._on_canvas_configure)
        c.bind("<ButtonPress-1>", self.on_button_press)
        c.bind("<B1-Motion>", self.on_move_press)
        c.bind("<ButtonRelease-1>", self.on_button_release)
        c.bind("<Double-Button-1>", self._on_double_click)
        c.bind("<Motion>", self._on_motion)

        if sys.platform == "darwin":
            c.bind("<ButtonPress-2>", self.on_right_click)
            c.bind("<ButtonPress-3>", self.on_middle_press)
            c.bind("<B3-Motion>", self.on_middle_drag)
            c.bind("<ButtonRelease-3>", self.on_middle_release)
            c.bind("<Control-ButtonPress-1>", self.on_right_click)
        else:
            c.bind("<ButtonPress-3>", self.on_right_click)
            c.bind("<ButtonPress-2>", self.on_middle_press)
            c.bind("<B2-Motion>", self.on_middle_drag)
            c.bind("<ButtonRelease-2>", self.on_middle_release)

        c.bind("<MouseWheel>", self._on_mousewheel)
        c.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        c.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        c.bind("<Button-4>", self._on_mousewheel_linux)
        c.bind("<Button-5>", self._on_mousewheel_linux)
        c.bind("<Control-Button-4>", self._on_ctrl_mousewheel_linux)
        c.bind("<Control-Button-5>", self._on_ctrl_mousewheel_linux)
        c.bind("<Shift-Button-4>", self._on_shift_mousewheel_linux)
        c.bind("<Shift-Button-5>", self._on_shift_mousewheel_linux)

    # ──────────────────────────────────────────────────────────────────────────
    #  Coordinate conversion
    # ──────────────────────────────────────────────────────────────────────────
    def canvas_to_image(self, cx, cy):
        ix = (cx - self.offset_x) / self.scale
        iy = (cy - self.offset_y) / self.scale
        return ix, iy

    def image_to_canvas(self, ix, iy):
        cx = ix * self.scale + self.offset_x
        cy = iy * self.scale + self.offset_y
        return cx, cy

    # ──────────────────────────────────────────────────────────────────────────
    #  Load image
    # ──────────────────────────────────────────────────────────────────────────
    def load_image(self):
        a = self.app
        if not a.images or not a.image_folder:
            return
        if a.index >= len(a.images):
            a.index = 0
        if a.index < 0:
            a.index = len(a.images) - 1
        print(f"[YoloLabeler] Loading image {a.index + 1}/{len(a.images)}: {a.images[a.index]}")

        a.boxes = []
        a.polygons = []
        a.box_authors = []
        a.polygon_authors = []
        self._invalidate_poly_bboxes()
        a.current_polygon = []
        a._undo_stack = []
        a._redo_stack = []
        a._vertex_redo_stack = []
        a._dragging_vertex = None
        a._drag_orig_pos = None
        self._poly_preview_line = None
        self._snap_indicator_item = None
        a._stream_active = False
        a._last_stream_pos = None
        a._selected_polygon_idx = None
        a._hovered_polygon_idx = None
        a._annotate_pred_reference = None
        a.start_x = None
        a.start_y = None
        a.rect = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._cached_scale = None
        self._cached_tk_image = None

        a._image_start_time = None

        img_path = os.path.join(a.image_folder, a.images[a.index])

        # Try loading the image; skip corrupt files
        attempts = 0
        while attempts < len(a.images):
            img_path = os.path.join(a.image_folder, a.images[a.index])
            try:
                a.original_image = Image.open(img_path)
                a.original_image.load()
                a.original_image = auto_orient_image(a.original_image)
                break
            except Exception as e:
                from tkinter import messagebox
                messagebox.showwarning(
                    "Image Error",
                    f"Could not load:\n{img_path}\n\n{e}")
                a.index += 1
                if a.index >= len(a.images):
                    a.index = 0
                attempts += 1
        else:
            from tkinter import messagebox
            messagebox.showerror(
                "No Valid Images",
                "No loadable images found in this folder.")
            return

        a.img_width, a.img_height = a.original_image.size
        a._image_dims[a.images[a.index]] = (a.img_width, a.img_height)

        self._initial_fit()
        self._load_existing_labels()
        img_name = a.images[a.index]
        if img_name not in a._session_loaded_counts:
            a._session_loaded_counts[img_name] = (
                len(a.boxes) + len(a.polygons))
        if not getattr(a, '_defer_display', False):
            self.display_image()
        a.update_title()
        a._update_status()

    def _initial_fit(self):
        a = self.app
        if a.img_width <= 0 or a.img_height <= 0:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10:
            cw = 1200
        if ch < 10:
            ch = 750
        sx = cw / a.img_width
        sy = ch / a.img_height
        fit_scale = min(sx, sy)
        self.zoom_index = self._nearest_zoom_index(fit_scale)
        self.scale = self.zoom_levels[self.zoom_index]
        self.offset_x = (cw - a.img_width * self.scale) / 2
        self.offset_y = (ch - a.img_height * self.scale) / 2

    def _nearest_zoom_index(self, target_scale):
        best_idx = 0
        best_diff = abs(self.zoom_levels[0] - target_scale)
        for i, level in enumerate(self.zoom_levels):
            diff = abs(level - target_scale)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    # ──────────────────────────────────────────────────────────────────────────
    #  Load existing YOLO labels
    # ──────────────────────────────────────────────────────────────────────────
    def _load_existing_labels(self):
        a = self.app
        if not a.detect_dir or not a.segment_dir:
            return
        stem = os.path.splitext(a.images[a.index])[0]

        detect_path = os.path.join(a.detect_dir, f"{stem}.txt")
        try:
            boxes, det_cids = parse_detect_labels(
                detect_path, a.img_width, a.img_height)
            a.boxes.extend(boxes)
            for cid in det_cids:
                if cid not in a.class_names:
                    a.class_names[cid] = f"class_{cid}"
                    a._refresh_class_dropdown()
                    a._save_classes_file()
        except Exception as e:
            print(f"Warning: Could not load detect labels for {stem}: {e}")

        segment_path = os.path.join(a.segment_dir, f"{stem}.txt")
        try:
            polygons, seg_cids = parse_segment_labels(
                segment_path, a.img_width, a.img_height)
            a.polygons.extend(polygons)
            self._invalidate_poly_bboxes()
            for cid in seg_cids:
                if cid not in a.class_names:
                    a.class_names[cid] = f"class_{cid}"
                    a._refresh_class_dropdown()
                    a._save_classes_file()
        except Exception as e:
            print(f"Warning: Could not load segment labels for {stem}: {e}")

        # Load per-annotation author metadata from annotation_stats.json
        a._load_annotation_authors()

    def _load_predictions(self, image_name, img_w, img_h):
        """Delegate to app-level shared helper."""
        return self.app._load_predictions(image_name, img_w, img_h)

    # ──────────────────────────────────────────────────────────────────────────
    #  Canvas resize debounce
    # ──────────────────────────────────────────────────────────────────────────
    def _on_canvas_configure(self, event=None):
        a = self.app
        if a.original_image is None:
            return
        self._cached_scale = None
        self._fast_resample = True
        self._initial_fit()
        self._request_redraw()
        if self._resize_after_id is not None:
            a.root.after_cancel(self._resize_after_id)
        self._resize_after_id = a.root.after(200, self._finalize_resize)

    def _finalize_resize(self):
        self._resize_after_id = None
        self._fast_resample = False
        self._cached_scale = None
        self._initial_fit()
        self.display_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Display (throttled)
    # ──────────────────────────────────────────────────────────────────────────
    def _request_redraw(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.app.root.after_idle(self._do_redraw)

    def _do_redraw(self):
        self._redraw_pending = False
        self.display_image()

    def display_image(self):
        self.render()

    def draw_help_overlay(self):
        self.render_help()

    def toggle_help(self, event=None):
        a = self.app
        if a.tabview.get() == "Review":
            a._review_show_help = not a._review_show_help
            a._review_tab._display_review_image()
        else:
            a.show_help = not a.show_help
            self.display_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Scroll / Zoom / Pan
    # ──────────────────────────────────────────────────────────────────────────
    def _on_mousewheel(self, event):
        delta = event.delta
        if sys.platform == "darwin":
            self.offset_y += delta * 2
        else:
            self.offset_y += (delta // 120) * 40
        self._request_redraw()

    def _on_shift_mousewheel(self, event):
        delta = event.delta
        if sys.platform == "darwin":
            self.offset_x += delta * 2
        else:
            self.offset_x += (delta // 120) * 40
        self._request_redraw()

    def _on_ctrl_mousewheel(self, event):
        direction = 1 if event.delta > 0 else -1
        self._zoom_step(event.x, event.y, direction)

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.offset_y += 40
        elif event.num == 5:
            self.offset_y -= 40
        self._request_redraw()

    def _on_shift_mousewheel_linux(self, event):
        if event.num == 4:
            self.offset_x += 40
        elif event.num == 5:
            self.offset_x -= 40
        self._request_redraw()

    def _on_ctrl_mousewheel_linux(self, event):
        direction = 1 if event.num == 4 else -1
        self._zoom_step(event.x, event.y, direction)

    def _zoom_step(self, cx, cy, direction):
        ix, iy = self.canvas_to_image(cx, cy)
        new_index = self.zoom_index + direction
        new_index = max(0, min(new_index, len(self.zoom_levels) - 1))
        if new_index == self.zoom_index:
            return
        self.zoom_index = new_index
        self.scale = self.zoom_levels[self.zoom_index]
        self.offset_x = cx - ix * self.scale
        self.offset_y = cy - iy * self.scale
        self._request_redraw()
        self.app._update_status()

    def on_middle_press(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.pan_start_offset_x = self.offset_x
        self.pan_start_offset_y = self.offset_y
        self.canvas.config(cursor="fleur")

    def on_middle_drag(self, event):
        if self.pan_start_x is None:
            return
        self.offset_x = self.pan_start_offset_x + (
            event.x - self.pan_start_x)
        self.offset_y = self.pan_start_offset_y + (
            event.y - self.pan_start_y)
        self._request_redraw()

    def on_middle_release(self, event):
        self.pan_start_x = None
        self.pan_start_y = None
        self.canvas.config(cursor="cross")

    # ──────────────────────────────────────────────────────────────────────────
    #  Mouse motion
    # ──────────────────────────────────────────────────────────────────────────
    def _on_motion(self, event):
        a = self.app
        self._mouse_canvas_x = event.x
        self._mouse_canvas_y = event.y

        # Streaming: auto-place vertices as mouse moves
        if (a.mode == "polygon" and a._stream_mode
                and a._stream_active and a.current_polygon):
            ix, iy = self.canvas_to_image(event.x, event.y)
            ix = max(0, min(a.img_width, ix))
            iy = max(0, min(a.img_height, iy))
            snapped_ix, snapped_iy = self._snap_to_edge(ix, iy)
            if (snapped_ix, snapped_iy) != (ix, iy):
                if (a.current_polygon[-1] != (snapped_ix, snapped_iy)):
                    a.current_polygon.append((snapped_ix, snapped_iy))
                    a._last_stream_pos = (snapped_ix, snapped_iy)
                    self.display_image()
            elif a._last_stream_pos:
                dist = math.hypot(
                    ix - a._last_stream_pos[0],
                    iy - a._last_stream_pos[1])
                if dist >= STREAM_MIN_DISTANCE:
                    a.current_polygon.append((ix, iy))
                    a._last_stream_pos = (ix, iy)
                    self.display_image()

        # Throttle expensive hover/snap checks (~60fps cap)
        now = time.monotonic()
        _motion_throttled = (now - self._motion_last_time) < 0.016

        # Update snap indicator
        if not _motion_throttled and a.mode == "polygon" and a.snap_enabled:
            ix, iy = self.canvas_to_image(event.x, event.y)
            snapped = self._maybe_snap(ix, iy)
            if snapped != (ix, iy):
                sx, sy = self.image_to_canvas(*snapped)
                snap_r = 12
                if self._snap_indicator_item:
                    try:
                        self.canvas.coords(
                            self._snap_indicator_item,
                            sx - snap_r, sy - snap_r, sx + snap_r, sy + snap_r)
                    except tk.TclError:
                        self._snap_indicator_item = None
                if not self._snap_indicator_item:
                    self._snap_indicator_item = self.canvas.create_oval(
                        sx - snap_r, sy - snap_r, sx + snap_r, sy + snap_r,
                        outline="#FFE7B1", fill="#FFE7B1",
                        width=2, stipple="gray50")
            else:
                if self._snap_indicator_item:
                    try:
                        self.canvas.delete(self._snap_indicator_item)
                    except tk.TclError:
                        pass
                    self._snap_indicator_item = None
        elif not (a.mode == "polygon" and a.snap_enabled):
            if self._snap_indicator_item:
                try:
                    self.canvas.delete(self._snap_indicator_item)
                except tk.TclError:
                    pass
                self._snap_indicator_item = None

        # Polygon hover detection
        if not _motion_throttled and a.mode == "polygon":
            self._motion_last_time = now
            ix, iy = self.canvas_to_image(event.x, event.y)
            new_hover = None
            hover_thr = 25
            if not a.current_polygon:
                vhit = self._find_nearest_vertex(event.x, event.y, threshold=hover_thr)
                if vhit:
                    new_hover = vhit[0]
                else:
                    ehit = self._find_nearest_edge_selected(event.x, event.y, threshold=hover_thr)
                    if ehit is not None:
                        new_hover = ehit
                    else:
                        for pi, (points, _) in enumerate(a.polygons):
                            if self._point_in_polygon(ix, iy, points):
                                new_hover = pi
                                break
            else:
                vhit = self._find_nearest_vertex(event.x, event.y, threshold=hover_thr + 5)
                if vhit:
                    new_hover = vhit[0]
                else:
                    ehit = self._find_nearest_edge_selected(event.x, event.y, threshold=hover_thr)
                    if ehit is not None:
                        new_hover = ehit
                    else:
                        for pi, (points, _) in enumerate(a.polygons):
                            if self._point_in_polygon(ix, iy, points):
                                new_hover = pi
                                break
            if new_hover != a._hovered_polygon_idx:
                a._hovered_polygon_idx = new_hover
                self._request_redraw()
        elif a.mode != "polygon":
            if a._hovered_polygon_idx is not None:
                a._hovered_polygon_idx = None

        # Polygon preview line
        if a.mode == "polygon" and a.current_polygon:
            if self._poly_preview_line is not None:
                try:
                    last_cx, last_cy = self.image_to_canvas(
                        *a.current_polygon[-1])
                    self.canvas.coords(
                        self._poly_preview_line,
                        last_cx, last_cy, event.x, event.y)
                except tk.TclError:
                    pass

    # ──────────────────────────────────────────────────────────────────────────
    #  Vertex snapping
    # ──────────────────────────────────────────────────────────────────────────
    def _maybe_snap(self, ix, iy, exclude=None):
        a = self.app
        if not a.snap_enabled:
            return (ix, iy)
        self._ensure_poly_bboxes()
        cx, cy = self.image_to_canvas(ix, iy)
        img_thr = SNAP_RADIUS / self.scale if self.scale > 0 else 1e9
        best_dist = SNAP_RADIUS
        best_pt = None
        for pidx, (points, _) in enumerate(a.polygons):
            if pidx < len(a._poly_bboxes):
                bx1, by1, bx2, by2 = a._poly_bboxes[pidx]
                if (ix + img_thr < bx1 or ix - img_thr > bx2
                        or iy + img_thr < by1 or iy - img_thr > by2):
                    continue
            for vidx, (px, py) in enumerate(points):
                if exclude is not None and (pidx, vidx) == exclude:
                    continue
                pcx, pcy = self.image_to_canvas(px, py)
                dist = math.hypot(cx - pcx, cy - pcy)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = (px, py)
        if best_pt:
            return best_pt
        return (ix, iy)

    def _snap_to_edge(self, ix, iy):
        """Snap to the nearest polygon edge in canvas space."""
        a = self.app
        if not a.snap_enabled:
            return (ix, iy)
        self._ensure_poly_bboxes()
        cx, cy = self.image_to_canvas(ix, iy)
        img_thr = SNAP_RADIUS / self.scale if self.scale > 0 else 1e9
        best_dist = SNAP_RADIUS
        best_pt = None
        for pidx, (points, _) in enumerate(a.polygons):
            if pidx < len(a._poly_bboxes):
                bx1, by1, bx2, by2 = a._poly_bboxes[pidx]
                if (ix + img_thr < bx1 or ix - img_thr > bx2
                        or iy + img_thr < by1 or iy - img_thr > by2):
                    continue
            n = len(points)
            for ei in range(n):
                ax, ay = self.image_to_canvas(*points[ei])
                bx, by = self.image_to_canvas(*points[(ei + 1) % n])
                dx, dy = bx - ax, by - ay
                len_sq = dx * dx + dy * dy
                if len_sq == 0:
                    continue
                t = max(0.0, min(1.0,
                        ((cx - ax) * dx + (cy - ay) * dy) / len_sq))
                proj_cx = ax + t * dx
                proj_cy = ay + t * dy
                dist = math.hypot(cx - proj_cx, cy - proj_cy)
                if dist < best_dist:
                    best_dist = dist
                    pix, piy = self.canvas_to_image(proj_cx, proj_cy)
                    pix = max(0, min(a.img_width, pix))
                    piy = max(0, min(a.img_height, piy))
                    best_pt = (pix, piy)
        if best_pt:
            return best_pt
        return (ix, iy)

    # ──────────────────────────────────────────────────────────────────────────
    #  Mouse event dispatch
    # ──────────────────────────────────────────────────────────────────────────
    def on_button_press(self, event):
        if self.app.mode == "box":
            self._box_press(event)
        else:
            self._poly_press(event)

    def on_move_press(self, event):
        if self.app.mode == "box":
            self._box_drag(event)
        else:
            self._poly_drag(event)

    def on_button_release(self, event):
        if self.app.mode == "box":
            self._box_release(event)
        else:
            self._poly_release(event)

    def _on_double_click(self, event):
        a = self.app
        if a.mode != "polygon":
            return
        if a.current_polygon:
            a._stream_active = False
            a._last_stream_pos = None
            if len(a.current_polygon) >= 3:
                self._close_polygon()
            return

    def _clear_drag_state(self):
        self.engine.clear_drag_state()
        self.canvas.config(cursor="cross")

    def on_right_click(self, event):
        a = self.app
        if a.mode == "polygon" and a.current_polygon:
            a.current_polygon = []
            a._stream_active = False
            a._last_stream_pos = None
            self.display_image()
            return

        click_ix, click_iy = self.canvas_to_image(event.x, event.y)

        if a.mode == "polygon" and a._selected_polygon_idx is not None:
            pi = a._selected_polygon_idx
            if pi < len(a.polygons):
                vertex_hit = self._find_nearest_vertex(
                    event.x, event.y, threshold=10)
                if vertex_hit and vertex_hit[0] == pi:
                    vi = vertex_hit[1]
                    points, cls = a.polygons[pi]
                    self._push_undo()
                    if len(points) <= 3:
                        a.polygons.pop(pi)
                        a.polygon_authors.pop(pi)
                        a._selected_polygon_idx = None
                    else:
                        new_pts = list(points)
                        new_pts.pop(vi)
                        a.polygons[pi] = (new_pts, cls)
                    self._invalidate_poly_bboxes()
                    self._clear_drag_state()
                    a._mark_image_annotated()
                    self.display_image()
                    a.update_title()
                    return
                if self._point_in_polygon(click_ix, click_iy,
                                          a.polygons[pi][0]):
                    self._push_undo()
                    a.polygons.pop(pi)
                    a.polygon_authors.pop(pi)
                    self._invalidate_poly_bboxes()
                    a._selected_polygon_idx = None
                    self._clear_drag_state()
                    a._mark_image_annotated()
                    self.display_image()
                    a.update_title()
                    return
            a._selected_polygon_idx = None
            self.display_image()
            if a._review_return_pending:
                a.root.after(50, a._review_tab._review_confirm_dialog)
            return

        for i, (x1, y1, x2, y2, _) in enumerate(a.boxes):
            if x1 <= click_ix <= x2 and y1 <= click_iy <= y2:
                self._push_undo()
                a.boxes.pop(i)
                a.box_authors.pop(i)
                a._mark_image_annotated()
                self.display_image()
                a.update_title()
                return

        for i, (points, _) in enumerate(a.polygons):
            if self._point_in_polygon(click_ix, click_iy, points):
                self._push_undo()
                self._clear_drag_state()
                a.polygons.pop(i)
                a.polygon_authors.pop(i)
                self._invalidate_poly_bboxes()
                a._selected_polygon_idx = None
                a._mark_image_annotated()
                self.display_image()
                a.update_title()
                return

    # ──────────────────────────────────────────────────────────────────────────
    #  Box mode
    # ──────────────────────────────────────────────────────────────────────────
    def _box_press(self, event):
        a = self.app
        a.start_x = event.x
        a.start_y = event.y
        color = a._get_class_color(a.active_class)
        a.rect = self.canvas.create_rectangle(
            a.start_x, a.start_y, a.start_x, a.start_y,
            outline=color, width=2)

    def _box_drag(self, event):
        a = self.app
        if (a.rect and a.start_x is not None
                and a.start_y is not None):
            self.canvas.coords(
                a.rect, a.start_x, a.start_y, event.x, event.y)

    def _box_release(self, event):
        a = self.app
        if a.start_x is None or a.start_y is None:
            return
        ix1, iy1 = self.canvas_to_image(a.start_x, a.start_y)
        ix2, iy2 = self.canvas_to_image(event.x, event.y)

        x1 = max(0, min(ix1, ix2))
        y1 = max(0, min(iy1, iy2))
        x2 = min(a.img_width, max(ix1, ix2))
        y2 = min(a.img_height, max(iy1, iy2))

        if (x2 - x1) < 3 or (y2 - y1) < 3:
            if a.rect:
                self.canvas.delete(a.rect)
            a.rect = None
            return

        self._push_undo()
        a.boxes.append((x1, y1, x2, y2, a.active_class))
        a.box_authors.append(a._current_user)
        a._mark_image_annotated()
        a._record_annotation_added()
        a.rect = None
        self.display_image()
        a.update_title()

        if a._review_return_pending:
            a.root.after(50, a._review_tab._review_confirm_dialog)

    # ──────────────────────────────────────────────────────────────────────────
    #  Polygon mode
    # ──────────────────────────────────────────────────────────────────────────
    def _poly_press(self, event):
        a = self.app
        ix, iy = self.canvas_to_image(event.x, event.y)

        if a.current_polygon:
            if a._stream_mode:
                if a._stream_active:
                    a._stream_active = False
                    a._last_stream_pos = None
                    self.display_image()
                else:
                    snapped = self._maybe_snap(ix, iy)
                    ix, iy = snapped
                    ix = max(0, min(a.img_width, ix))
                    iy = max(0, min(a.img_height, iy))
                    a.current_polygon.append((ix, iy))
                    a._stream_active = True
                    a._last_stream_pos = (ix, iy)
                    a._vertex_redo_stack.clear()
                    self.display_image()
            else:
                snapped = self._maybe_snap(ix, iy)
                ix, iy = snapped
                ix = max(0, min(a.img_width, ix))
                iy = max(0, min(a.img_height, iy))
                a.current_polygon.append((ix, iy))
                a._vertex_redo_stack.clear()
                self.display_image()
            return

        if a._selected_polygon_idx is not None:
            pi = a._selected_polygon_idx
            if pi < len(a.polygons):
                best_vi, best_vd = None, 8
                for vi, (px, py) in enumerate(a.polygons[pi][0]):
                    vcx, vcy = self.image_to_canvas(px, py)
                    d = math.hypot(event.x - vcx, event.y - vcy)
                    if d < best_vd:
                        best_vd = d
                        best_vi = vi
                if best_vi is not None:
                    self._push_undo()
                    a._dragging_vertex = (pi, best_vi)
                    a._drag_orig_pos = a.polygons[pi][0][best_vi]
                    self.canvas.config(cursor="fleur")
                    return
                best_ei, best_ed, best_ept = None, 6, None
                pts_sel = a.polygons[pi][0]
                n_sel = len(pts_sel)
                for ei in range(n_sel):
                    ax, ay = self.image_to_canvas(*pts_sel[ei])
                    bx, by = self.image_to_canvas(*pts_sel[(ei + 1) % n_sel])
                    d = point_to_segment_dist(event.x, event.y, ax, ay, bx, by)
                    if d < best_ed:
                        best_ed = d
                        edx, edy = bx - ax, by - ay
                        len_sq = edx * edx + edy * edy
                        if len_sq == 0:
                            proj_cx, proj_cy = ax, ay
                        else:
                            t = max(0.0, min(1.0, ((event.x - ax) * edx + (event.y - ay) * edy) / len_sq))
                            proj_cx = ax + t * edx
                            proj_cy = ay + t * edy
                        pix, piy = self.canvas_to_image(proj_cx, proj_cy)
                        pix = max(0, min(a.img_width, pix))
                        piy = max(0, min(a.img_height, piy))
                        best_ei = ei
                        best_ept = (pix, piy)
                if best_ei is not None:
                    self._push_undo()
                    points, cls = a.polygons[pi]
                    new_points = list(points)
                    new_points.insert(best_ei + 1, best_ept)
                    a.polygons[pi] = (new_points, cls)
                    self._invalidate_poly_bboxes()
                    a._dragging_vertex = (pi, best_ei + 1)
                    a._drag_orig_pos = best_ept
                    self.canvas.config(cursor="fleur")
                    self.display_image()
                    return
            just_deselected = True
            a._selected_polygon_idx = None
            if a._review_return_pending:
                self.display_image()
                a.root.after(50, a._review_tab._review_confirm_dialog)
                return
        else:
            just_deselected = False

        if not a.snap_enabled:
            vhit = self._find_nearest_vertex(event.x, event.y, threshold=15)
            if vhit:
                a._selected_polygon_idx = vhit[0]
                self.display_image()
                return
        for pi, (points, _) in enumerate(a.polygons):
            if self._point_in_polygon(ix, iy, points):
                a._selected_polygon_idx = pi
                self.display_image()
                return

        if just_deselected:
            self.display_image()
            return

        ix, iy = self._maybe_snap(ix, iy)
        ix = max(0, min(a.img_width, ix))
        iy = max(0, min(a.img_height, iy))
        a.current_polygon = [(ix, iy)]

        if a._stream_mode:
            a._stream_active = True
            a._last_stream_pos = (ix, iy)

        self.display_image()

    def _poly_drag(self, event):
        a = self.app
        if a._dragging_vertex is not None:
            pi, vi = a._dragging_vertex
            if pi >= len(a.polygons):
                self._clear_drag_state()
                return
            if pi != a._selected_polygon_idx:
                self._clear_drag_state()
                return
            raw_ix, raw_iy = self.canvas_to_image(event.x, event.y)
            ix, iy = self._maybe_snap(raw_ix, raw_iy, exclude=(pi, vi))
            did_snap = (ix, iy) != (raw_ix, raw_iy)
            ix = max(0, min(a.img_width, ix))
            iy = max(0, min(a.img_height, iy))
            points, cls = a.polygons[pi]
            new_points = list(points)
            new_points[vi] = (ix, iy)
            a.polygons[pi] = (new_points, cls)
            self._invalidate_poly_bboxes()
            self.display_image()
            if a.snap_enabled:
                if did_snap:
                    sx, sy = self.image_to_canvas(ix, iy)
                    snap_r = 12
                    if self._snap_indicator_item:
                        try:
                            self.canvas.coords(
                                self._snap_indicator_item,
                                sx - snap_r, sy - snap_r, sx + snap_r, sy + snap_r)
                        except tk.TclError:
                            self._snap_indicator_item = None
                    if not self._snap_indicator_item:
                        self._snap_indicator_item = self.canvas.create_oval(
                            sx - snap_r, sy - snap_r, sx + snap_r, sy + snap_r,
                            outline="#FFE7B1", fill="#FFE7B1",
                            width=2, stipple="gray50")
                else:
                    if self._snap_indicator_item:
                        try:
                            self.canvas.delete(self._snap_indicator_item)
                        except tk.TclError:
                            pass
                        self._snap_indicator_item = None

    def _poly_release(self, event):
        a = self.app
        if a._dragging_vertex is not None:
            a._mark_image_annotated()
            a._dragging_vertex = None
            a._drag_orig_pos = None
            self.canvas.config(cursor="cross")

    def _close_polygon(self):
        a = self.app
        added = self.engine.close_current_polygon()
        if not added:
            self.display_image()
            return
        a._mark_image_annotated()
        a._record_annotation_added()
        self._poly_preview_line = None
        self.display_image()
        a.update_title()

        if a._review_return_pending:
            a.root.after(50, a._review_tab._review_confirm_dialog)

    # ──────────────────────────────────────────────────────────────────────────
    #  Polygon spatial index
    # ──────────────────────────────────────────────────────────────────────────
    def _invalidate_poly_bboxes(self):
        self.engine.invalidate_poly_bboxes()

    def _ensure_poly_bboxes(self):
        self.engine.ensure_poly_bboxes()

    # ──────────────────────────────────────────────────────────────────────────
    #  Polygon geometry helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _find_nearest_vertex(self, cx, cy, threshold=8):
        a = self.app
        self._ensure_poly_bboxes()
        qix, qiy = self.canvas_to_image(cx, cy)
        img_thr = threshold / self.scale if self.scale > 0 else 1e9
        best = None
        best_dist = threshold
        for pi, (points, _) in enumerate(a.polygons):
            if pi < len(a._poly_bboxes):
                bx1, by1, bx2, by2 = a._poly_bboxes[pi]
                if (qix + img_thr < bx1 or qix - img_thr > bx2
                        or qiy + img_thr < by1 or qiy - img_thr > by2):
                    continue
            for vi, (px, py) in enumerate(points):
                vcx, vcy = self.image_to_canvas(px, py)
                dist = math.hypot(cx - vcx, cy - vcy)
                if dist < best_dist:
                    best_dist = dist
                    best = (pi, vi)
        return best

    def _find_nearest_edge(self, cx, cy, threshold=6):
        a = self.app
        self._ensure_poly_bboxes()
        qix, qiy = self.canvas_to_image(cx, cy)
        img_thr = threshold / self.scale if self.scale > 0 else 1e9
        best = None
        best_dist = threshold
        for pi, (points, _) in enumerate(a.polygons):
            if pi < len(a._poly_bboxes):
                bx1, by1, bx2, by2 = a._poly_bboxes[pi]
                if (qix + img_thr < bx1 or qix - img_thr > bx2
                        or qiy + img_thr < by1 or qiy - img_thr > by2):
                    continue
            n = len(points)
            for ei in range(n):
                ax, ay = self.image_to_canvas(*points[ei])
                bx, by = self.image_to_canvas(*points[(ei + 1) % n])
                dist = point_to_segment_dist(cx, cy, ax, ay, bx, by)
                if dist < best_dist:
                    best_dist = dist
                    ix, iy = self.canvas_to_image(cx, cy)
                    ix = max(0, min(a.img_width, ix))
                    iy = max(0, min(a.img_height, iy))
                    best = (pi, ei, (ix, iy))
        return best

    def _find_nearest_edge_selected(self, cx, cy, threshold=6):
        a = self.app
        pi = a._selected_polygon_idx
        if pi is None or pi >= len(a.polygons):
            return None
        points = a.polygons[pi][0]
        n = len(points)
        for ei in range(n):
            ax, ay = self.image_to_canvas(*points[ei])
            bx, by = self.image_to_canvas(*points[(ei + 1) % n])
            dist = point_to_segment_dist(cx, cy, ax, ay, bx, by)
            if dist < threshold:
                return pi
        return None

    _point_in_polygon = staticmethod(point_in_polygon)

    # ──────────────────────────────────────────────────────────────────────────
    #  Navigation
    # ──────────────────────────────────────────────────────────────────────────
    def next_image(self, event=None):
        a = self.app
        a._record_image_time()
        self.save_annotations()
        a._save_stats()
        if a._active_filter != "all" and a._filtered_indices:
            for idx in a._filtered_indices:
                if idx > a.index:
                    a.index = idx
                    self.load_image()
                    return
            a.index = a._filtered_indices[0]
            self.load_image()
            return
        a.index += 1
        if a.index >= len(a.images):
            a.index = 0
        self.load_image()

    def prev_image(self, event=None):
        a = self.app
        a._record_image_time()
        self.save_annotations()
        a._save_stats()
        if a._active_filter != "all" and a._filtered_indices:
            for idx in reversed(a._filtered_indices):
                if idx < a.index:
                    a.index = idx
                    self.load_image()
                    return
            a.index = a._filtered_indices[-1]
            self.load_image()
            return
        a.index -= 1
        if a.index < 0:
            a.index = len(a.images) - 1
        self.load_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Undo / Redo
    # ──────────────────────────────────────────────────────────────────────────
    def _push_undo(self):
        self.engine.push_undo()

    def undo_last(self, event=None):
        a = self.app
        if a.current_polygon:
            pt = a.current_polygon.pop()
            a._vertex_redo_stack.append(pt)
            if a.current_polygon:
                self.display_image()
                return
            self.display_image()
            if not a._undo_stack:
                return
        if self.engine.undo_snapshot():
            self.canvas.config(cursor="cross")
            self.display_image()
            a.update_title()

    def redo_last(self, event=None):
        a = self.app
        if a.current_polygon and a._vertex_redo_stack:
            pt = a._vertex_redo_stack.pop()
            a.current_polygon.append(pt)
            self.display_image()
            return
        if self.engine.redo_snapshot():
            self.canvas.config(cursor="cross")
            self.display_image()
            a.update_title()

    # ──────────────────────────────────────────────────────────────────────────
    #  Save annotations
    # ──────────────────────────────────────────────────────────────────────────
    def save_annotations(self):
        a = self.app
        if a.images:
            print(f"[YoloLabeler] Saving annotations for {a.images[a.index]} "
                  f"({len(a.boxes)} boxes, {len(a.polygons)} polygons)")
        self.engine.save()
        a._save_annotation_authors()

    # ══════════════════════════════════════════════════════════════════════════
    #  Rendering (absorbed from AnnotateRenderer)
    # ══════════════════════════════════════════════════════════════════════════
    def render(self):
        a = self.app
        if a.original_image is None:
            return
        canvas = self.canvas
        cw = canvas.winfo_width() or 1200
        ch = canvas.winfo_height() or 800

        vis_x1, vis_y1 = self.canvas_to_image(0, 0)
        vis_x2, vis_y2 = self.canvas_to_image(cw, ch)

        crop_x1 = max(0, int(vis_x1))
        crop_y1 = max(0, int(vis_y1))
        crop_x2 = min(a.img_width, int(vis_x2) + 1)
        crop_y2 = min(a.img_height, int(vis_y2) + 1)

        cache_key = (self.scale, crop_x1, crop_y1, crop_x2, crop_y2)

        if self._cached_scale != cache_key:
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            if crop_w > 0 and crop_h > 0:
                cropped = a.original_image.crop(
                    (crop_x1, crop_y1, crop_x2, crop_y2))
                out_w = max(int(crop_w * self.scale), 1)
                out_h = max(int(crop_h * self.scale), 1)
                resized = cropped.resize(
                    (out_w, out_h),
                    Image.Resampling.BILINEAR if self._fast_resample
                    else Image.Resampling.LANCZOS)
                self._cached_tk_image = ImageTk.PhotoImage(resized)
            else:
                self._cached_tk_image = None
            self._cached_scale = cache_key

        canvas.delete("all")
        self._snap_indicator_item = None

        if self._cached_tk_image is not None:
            place_x = self.offset_x + crop_x1 * self.scale
            place_y = self.offset_y + crop_y1 * self.scale
            canvas.create_image(
                place_x, place_y, anchor="nw", image=self._cached_tk_image)

        # Scale-dependent symbology
        s = self.scale
        line_w = max(1, min(2 + s * 0.5, 6))
        poly_w = max(1, min(2.5 + s * 0.5, 7))
        vert_r = max(3, min(VERTEX_HANDLE_RADIUS * (1.6 - s * 0.2), 12))
        sel_vert_r = max(vert_r + 2,
                         min(VERTEX_HANDLE_RADIUS * (2.2 - s * 0.2), 16))
        label_size = max(7, min(int(9 * (0.6 + s * 0.4)), 18))
        dash_a = max(2, int(4 * (0.5 + s * 0.5)))
        dash_b = max(2, int(4 * (0.5 + s * 0.5)))

        def _halo(x, y, text, fill, **kw):
            halo_text(canvas, x, y, text, fill, **kw)

        if a.mode == "box" and a._annotation_visible:
            for box in a.boxes:
                x1, y1, x2, y2, class_id = box
                if class_id != a.active_class:
                    continue
                if x2 < vis_x1 or x1 > vis_x2 or y2 < vis_y1 or y1 > vis_y2:
                    continue
                cx1, cy1 = self.image_to_canvas(x1, y1)
                cx2, cy2 = self.image_to_canvas(x2, y2)
                color = a._get_class_color(class_id)
                canvas.create_rectangle(
                    cx1, cy1, cx2, cy2, outline=color, width=line_w)
                class_name = a.class_names.get(class_id, str(class_id))
                _halo(cx1 + 2, cy1 - 2, anchor="sw",
                      text=f"{class_id}: {class_name}",
                      fill=color,
                      font=(a.font_family, label_size, "bold"))

        if a.mode == "polygon" and a._annotation_visible:
            for poly_idx, (points, class_id) in enumerate(a.polygons):
                is_selected = (poly_idx == a._selected_polygon_idx)
                if class_id != a.active_class and not is_selected:
                    continue
                if points and not is_selected:
                    pxs = [p[0] for p in points]
                    pys = [p[1] for p in points]
                    if (max(pxs) < vis_x1 or min(pxs) > vis_x2
                            or max(pys) < vis_y1 or min(pys) > vis_y2):
                        continue
                color = a._get_class_color(class_id)
                draw_color = "#00BFFF" if is_selected else color
                canvas_pts = []
                for px, py in points:
                    cx, cy = self.image_to_canvas(px, py)
                    canvas_pts.extend([cx, cy])
                if len(canvas_pts) >= 6:
                    canvas.create_polygon(
                        *canvas_pts, outline=draw_color, fill="",
                        width=poly_w)
                show_verts = (
                    is_selected
                    or poly_idx == a._hovered_polygon_idx
                    or (a._dragging_vertex is not None
                        and a._dragging_vertex[0] == poly_idx)
                )
                if (not show_verts
                        and a._dragging_vertex is not None
                        and a.snap_enabled
                        and points):
                    dpi = a._dragging_vertex[0]
                    if dpi < len(a.polygons):
                        dvi = a._dragging_vertex[1]
                        dpts = a.polygons[dpi][0]
                        if dvi < len(dpts):
                            dcx, dcy = self.image_to_canvas(*dpts[dvi])
                            for px, py in points:
                                pcx, pcy = self.image_to_canvas(px, py)
                                if math.hypot(dcx - pcx, dcy - pcy) \
                                        < SNAP_RADIUS * 3:
                                    show_verts = True
                                    break
                if show_verts:
                    r = sel_vert_r if is_selected else vert_r
                    for px, py in points:
                        cx, cy = self.image_to_canvas(px, py)
                        canvas.create_oval(
                            cx - r, cy - r, cx + r, cy + r,
                            fill=draw_color, outline="white", width=1)
                if points:
                    lx, ly = self.image_to_canvas(*points[0])
                    class_name = a.class_names.get(class_id, str(class_id))
                    _halo(lx + 2, ly - 2, anchor="sw",
                          text=f"{class_id}: {class_name}",
                          fill=draw_color,
                          font=(a.font_family, label_size, "bold"))

        if a.current_polygon:
            color = a._get_class_color(a.active_class)
            for i, (px, py) in enumerate(a.current_polygon):
                cx, cy = self.image_to_canvas(px, py)
                canvas.create_oval(
                    cx - vert_r, cy - vert_r,
                    cx + vert_r, cy + vert_r,
                    fill=color, outline="white", width=1)
                if i > 0:
                    prev_cx, prev_cy = self.image_to_canvas(
                        *a.current_polygon[i - 1])
                    canvas.create_line(
                        prev_cx, prev_cy, cx, cy,
                        fill=color, width=line_w, dash=(dash_a, dash_b))
            last_cx, last_cy = self.image_to_canvas(
                *a.current_polygon[-1])
            self._poly_preview_line = canvas.create_line(
                last_cx, last_cy,
                self._mouse_canvas_x, self._mouse_canvas_y,
                fill=color, width=max(1, line_w * 0.5),
                dash=(dash_a // 2 or 1, dash_b))

        # Prediction reference overlay
        if a._annotate_pred_reference:
            ref = a._annotate_pred_reference
            PRED_REF_COLOR = "#00BFFF"
            ref_dash = (6, 4)
            ref_lw = max(1, min(2 + s * 0.5, 5))
            ref_label_size = max(8, min(int(10 * (0.6 + s * 0.4)), 16))
            cid = ref.get('class_id', 0)
            conf = ref.get('conf', 0)
            name = a.class_names.get(cid, str(cid))

            if ref['type'] == 'box':
                x1, y1, x2, y2 = ref['coords']
                cx1, cy1 = self.image_to_canvas(x1, y1)
                cx2, cy2 = self.image_to_canvas(x2, y2)
                canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline=PRED_REF_COLOR, width=ref_lw,
                    dash=ref_dash)
                _halo(cx1 + 2, cy1 - 2, anchor="sw",
                      text=f"Pred {cid}: {name} ({conf:.2f})",
                      fill=PRED_REF_COLOR,
                      font=(a.font_family, ref_label_size, "bold"))
            elif ref['type'] == 'polygon':
                pts = ref['coords']
                canvas_pts = []
                for px_pt, py_pt in pts:
                    cx_p, cy_p = self.image_to_canvas(px_pt, py_pt)
                    canvas_pts.extend([cx_p, cy_p])
                if len(canvas_pts) >= 6:
                    canvas.create_polygon(
                        *canvas_pts, outline=PRED_REF_COLOR,
                        fill="", width=ref_lw, dash=ref_dash)
                if pts:
                    lx, ly = self.image_to_canvas(*pts[0])
                    _halo(lx + 2, ly - 2, anchor="sw",
                          text=f"Pred {cid}: {name} ({conf:.2f})",
                          fill=PRED_REF_COLOR,
                          font=(a.font_family, ref_label_size, "bold"))

        self.render_help()

    def render_help(self):
        a = self.app
        if not a.show_help:
            return

        canvas = self.canvas

        if a.mode == "box":
            help_lines = [
                "\u2500\u2500 Keyboard \u2500\u2500",
                "  h              Toggle this help",
                "  m              Toggle Box / Polygon mode",
                "  0-9            Select class by ID",
                "  Ctrl+Y         Redo",
                "  Ctrl+Z         Undo",
                "  \u2190 / \u2192          Previous / Next image",
                "",
                "\u2500\u2500 Mouse \u2500\u2500",
                "  Ctrl+Scroll          Zoom at cursor",
                "  Left-click + drag    Draw bounding box",
                "  Middle-click         Pan (drag)",
                "  Right-click          Delete annotation",
                "  Scroll               Pan up / down",
                "  Shift+Scroll         Pan left / right",
            ]
        else:
            help_lines = [
                "\u2500\u2500 Keyboard \u2500\u2500",
                "  h              Toggle this help",
                "  m              Toggle Box / Polygon mode",
                "  s              Toggle vertex snapping",
                "  v              Toggle vertex streaming",
                "  0-9            Select class by ID",
                "  Ctrl+Y         Redo",
                "  Ctrl+Z         Undo",
                "  Escape         Cancel / Deselect polygon",
                "  \u2190 / \u2192          Previous / Next image",
                "",
                "\u2500\u2500 Mouse \u2500\u2500",
                "  Click edge           Insert vertex (selected polygon)",
                "  Ctrl+Scroll          Zoom at cursor",
                "  Double-click         Close polygon",
                "  Drag vertex          Move vertex (selected polygon)",
                "  Left-click           Place vertex / Select polygon",
                "  Middle-click         Pan (drag)",
                "  Right-click          Delete annotation / vertex",
                "  Scroll               Pan up / down",
                "  Shift+Scroll         Pan left / right",
                "",
                "\u2500\u2500 Streaming \u2500\u2500",
                "  Press 'v' to enable stream mode, then:",
                "  Click to start streaming vertices,",
                "  move mouse to trace, click to pause,",
                "  double-click or Escape to finish.",
            ]

        if sys.platform == "darwin":
            font_family = "Menlo"
        else:
            font_family = "Consolas"
        font_size = 14
        pad = 14

        fnt = tkFont.Font(family=font_family, size=font_size)
        line_height = fnt.metrics("linespace") + 2
        max_text_w = max(fnt.measure(ln) for ln in help_lines) \
            if help_lines else 100

        block_w = max_text_w + pad * 3
        block_h = len(help_lines) * line_height + pad * 2
        x0, y0 = 10, 10

        canvas.create_rectangle(
            x0, y0, x0 + block_w, y0 + block_h,
            fill="#1A1A1A", outline="#444444", width=1, stipple="")

        for i, line in enumerate(help_lines):
            canvas.create_text(
                x0 + pad, y0 + pad + i * line_height,
                anchor="nw", text=line,
                fill=FG_COLOR, font=(font_family, font_size))
