"""AnnotateTab: annotation canvas, interaction and rendering for the Annotate tab.

Covers canvas construction, bindings, coordinate conversion, pan/zoom, image
loading, box/polygon interaction, snapping, vertex streaming, undo/redo, save,
and canvas rendering.
"""

import math
import os
import sys
import time
import tkinter as tk
import tkinter.font as tkFont

from PIL import Image, ImageTk

from yololabeler import keybindings
from yololabeler.annotation.document import load_document
from yololabeler.matching import point_to_segment_dist, point_in_polygon
from yololabeler.rendering import place_label
from yololabeler.review.engine import build_queue
from yololabeler.review.layer import (
    FLAG_COLOR, FLAG_MARK, SELECTION_COLOR, STATUS_COLORS, LayerStyle, draw_prediction_layer,
    status_color, status_colors_active,
)
from yololabeler.utils import auto_orient_image

# Interaction constants, carried over from the original implementation
VERTEX_HANDLE_RADIUS = 4      # base vertex marker radius in canvas px
BOX_CORNER_HIT_RADIUS = 8     # canvas px; matches _find_nearest_vertex's default tolerance
BOX_EDGE_HIT_RADIUS = 8       # canvas px from a box outline that selects or drags it; same as corners, provisional
MIN_BOX_SIDE = 3              # image px; the minimum box side enforced when drawing
# Streaming and snapping values taken from TCIP Agent's AnnotateTab.tsx
STREAM_MIN_DISTANCE = 6       # canvas px the pointer moves before the next streamed vertex
SNAP_RADIUS = 15              # canvas px radius for snapping to an existing vertex
SNAP_INDICATOR_RADIUS = 7     # canvas px radius of the dashed snap-target ring
SNAP_INDICATOR_COLOR = "#FFE7B1"
CLICK_SLOP = 3                # canvas px a press may move and still count as a click; provisional
FG_COLOR = "#E0E0E0"
CANVAS_BG = "#2D2D2D"
LEGEND_BG = "#1A1A1A"
LEGEND_BORDER = "#444444"


def _clamp_delta(delta, lo, hi):
    """delta held within [lo, hi]; unclamped when lo exceeds hi, a shape too large to fit."""
    if lo > hi:
        return delta
    return max(lo, min(hi, delta))


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

        # Legend chip in the lower left of the canvas
        self._legend_open = False
        self._legend_bbox = None
        self._legend_press = False

        # A press on a selected polygon's vertex: a drag moves it, a click starts a polygon there
        self._vertex_press = None
        self._vertex_drag_started = False

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

        a.document = None
        self._invalidate_poly_bboxes()
        a.current_polygon = []
        a._undo_stack = []
        a._redo_stack = []
        a._vertex_redo_stack = []
        a._dragging_vertex = None
        a._drag_orig_pos = None
        self._clear_box_edit_state()
        self._poly_preview_line = None
        self._snap_indicator_item = None
        a._stream_active = False
        a._last_stream_pos = None
        a._selected_annotation_id = None
        a._hovered_annotation_id = None
        a.start_x = None
        a.start_y = None
        a.rect = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._cached_scale = None
        self._cached_tk_image = None

        a._image_start_time = None

        # Try loading the image; skip corrupt files
        skipped = []
        attempts = 0
        while attempts < len(a.images):
            img_path = os.path.join(a.image_folder, a.images[a.index])
            try:
                a.original_image = Image.open(img_path)
                a.original_image.load()
                a.original_image = auto_orient_image(a.original_image)
                break
            except Exception as e:
                skipped.append(f"{a.images[a.index]} ({e})")
                a.index += 1
                if a.index >= len(a.images):
                    a.index = 0
                attempts += 1
        else:
            a.show_canvas_message("No loadable images found in this folder.")
            return

        a.img_width, a.img_height = a.original_image.size
        a._image_start_time = time.time()

        self.fit_to_window()
        rejected, sidecar_moved = self.load_document_for_current_image()
        a.load_errors = rejected
        messages = []
        if skipped:
            messages.append(f"{len(skipped)} images could not be opened and were "
                            f"skipped ({'; '.join(skipped)}).")
        if sidecar_moved:
            messages.append(f"This image's sidecar could not be read and was moved to "
                            f"{os.path.basename(sidecar_moved)}. Its annotations keep their "
                            f"geometry but lose their authors and provenance.")
        if rejected:
            messages.append(f"{len(rejected)} label lines could not be read "
                            f"({'; '.join(rejected)}). This image will not be saved until they are fixed.")
        img_name = a.images[a.index]
        if img_name not in a._session_loaded_counts:
            a._session_loaded_counts[img_name] = len(a.document.annotations)
        a.verdicts = a._review.verdicts(img_name)
        a._review_panel.load_predictions_for_current_image()
        if a.predictions_rejected:
            messages.append(f"{len(a.predictions_rejected)} prediction lines could not be read "
                            f"({'; '.join(a.predictions_rejected)}).")
        if messages:
            a.banner_text = "\n".join(messages)
        a._review_panel.refresh(keep_focus=False)
        a._review_panel.focus_item(a._review_panel.first_unreviewed(), switch_class=False)
        a._review_panel.update_labels()
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

    def fit_to_window(self):
        """Fit the whole image into the canvas and redraw."""
        self._initial_fit()
        self._cached_scale = None
        self._request_redraw()

    def zoom_centered(self, scale):
        """Zoom to the zoom level nearest scale with the whole image centred in the canvas."""
        a = self.app
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        self.zoom_index = self._nearest_zoom_index(scale)
        self.scale = self.zoom_levels[self.zoom_index]
        self.offset_x = (cw - a.img_width * self.scale) / 2
        self.offset_y = (ch - a.img_height * self.scale) / 2
        self._cached_scale = None
        self._request_redraw()
        a._update_status()

    def zoom_to_bbox(self, x1, y1, x2, y2):
        """Zoom so the box fills one third of the canvas, centred (spec 4.3)."""
        a = self.app
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        det_w, det_h = max(x2 - x1, 1), max(y2 - y1, 1)
        target = min(cw / (det_w * 3), ch / (det_h * 3))
        self.zoom_index = self._nearest_zoom_index(target)
        self.scale = self.zoom_levels[self.zoom_index]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.offset_x = cw / 2 - cx * self.scale
        self.offset_y = ch / 2 - cy * self.scale
        self._cached_scale = None
        self._request_redraw()
        a._update_status()

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
    #  Load the document for the current image
    # ──────────────────────────────────────────────────────────────────────────
    def load_document_for_current_image(self):
        """Read label files plus sidecar into a.document.

        Returns (rejected-line messages, quarantined sidecar path or None).
        """
        a = self.app
        img_name = a.images[a.index]
        detect, segment, sidecar = self.engine.label_paths()
        legacy_authors = a._stats_store.pop_legacy_authors(img_name)
        a.document, rejected, sidecar_moved = load_document(
            img_name, a.img_width, a.img_height, detect, segment, sidecar,
            legacy_authors=legacy_authors)
        if legacy_authors is not None:
            a._save_stats()
        a._register_class_ids({ann.class_id for ann in a.document.annotations})
        self._invalidate_poly_bboxes()
        return rejected, sidecar_moved

    # ── Visibility and selection ──────────────────────────────────────────────

    def visible_annotations(self):
        """Annotations drawn right now, in draw order; hit-testing uses the same list."""
        a = self.app
        if a.document is None or not a._annotation_visible:
            return []
        focus_pair = a.queue[a.queue_index].annotation if (
            a.queue and 0 <= a.queue_index < len(a.queue)) else None
        out = []
        for ann in a.document.annotations:
            if ann.id == a._selected_annotation_id or (focus_pair and ann.id == focus_pair.id):
                out.append(ann)
            elif ann.kind == a.mode and ann.class_id == a.active_class:
                out.append(ann)
        return out

    def _alive(self, ann_id):
        """True when ann_id still names an annotation of the current document."""
        doc = self.app.document
        if ann_id is None or doc is None:
            return False
        return any(ann.id == ann_id for ann in doc.annotations)

    def select_annotation(self, ann_id):
        """Select an annotation by id, switching the annotate mode to match its kind."""
        a = self.app
        if ann_id is not None:
            a._set_mode(a.document.get(ann_id).kind)
        a._selected_annotation_id = ann_id
        self.display_image()

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

        # Streaming: between the start and pause clicks, lay a vertex each time the
        # pointer has moved STREAM_MIN_DISTANCE screen pixels from the last one.
        if (a.mode == "polygon" and a._stream_mode
                and a._stream_active and a.current_polygon):
            last_cx, last_cy = self.image_to_canvas(*a.current_polygon[-1])
            if math.hypot(event.x - last_cx, event.y - last_cy) >= STREAM_MIN_DISTANCE:
                ix, iy = self._clamp(*self._maybe_snap(*self.canvas_to_image(event.x, event.y)))
                if a.current_polygon[-1] != (ix, iy):
                    a.current_polygon.append((ix, iy))
                    a._last_stream_pos = (ix, iy)
                    self.display_image()

        # Throttle expensive hover/snap checks (~60fps cap)
        now = time.monotonic()
        _motion_throttled = (now - self._motion_last_time) < 0.016

        if not _motion_throttled:
            self._update_snap_indicator()

        # Polygon hover detection
        if not _motion_throttled and a.mode == "polygon":
            self._motion_last_time = now
            ix, iy = self.canvas_to_image(event.x, event.y)
            new_hover = None
            hover_thr = 25
            # Vertices get a wider hit radius while a polygon is being drawn.
            vertex_thr = hover_thr + 5 if a.current_polygon else hover_thr
            vhit = self._find_nearest_vertex(event.x, event.y, threshold=vertex_thr)
            if vhit:
                new_hover = vhit[0]
            else:
                ehit = self._find_nearest_edge_selected(event.x, event.y, threshold=hover_thr)
                if ehit is not None:
                    new_hover = ehit
                else:
                    for ann in self.visible_annotations():
                        if ann.kind != "polygon":
                            continue
                        if self._point_in_polygon(ix, iy, ann.points):
                            new_hover = ann.id
                            break
            if new_hover != a._hovered_annotation_id:
                a._hovered_annotation_id = new_hover
                self._request_redraw()
        elif a.mode != "polygon":
            if a._hovered_annotation_id is not None:
                a._hovered_annotation_id = None
            on_outline = self._box_at_outline(event.x, event.y) is not None
            cursor = "fleur" if on_outline else "cross"
            if self.canvas.cget("cursor") != cursor:
                self.canvas.config(cursor=cursor)

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
    def _clamp(self, ix, iy):
        """Clamp an image point to the image bounds."""
        a = self.app
        return max(0, min(a.img_width, ix)), max(0, min(a.img_height, iy))

    def _update_snap_indicator(self):
        """Ring the vertex a click at the pointer would snap to, or hide the ring."""
        a = self.app
        if a.mode != "polygon" or not a.snap_enabled or a.original_image is None:
            self._hide_snap_indicator()
            return
        ix, iy = self.canvas_to_image(self._mouse_canvas_x, self._mouse_canvas_y)
        exclude = a._dragging_vertex if self._vertex_drag_started else None
        snapped = self._maybe_snap(ix, iy, exclude=exclude)
        if snapped != (ix, iy):
            self._show_snap_indicator(*self.image_to_canvas(*snapped))
        else:
            self._hide_snap_indicator()

    def _show_snap_indicator(self, sx, sy):
        """Place (or move) the dashed snap-target ring at canvas point (sx, sy)."""
        bbox = (sx - SNAP_INDICATOR_RADIUS, sy - SNAP_INDICATOR_RADIUS,
                sx + SNAP_INDICATOR_RADIUS, sy + SNAP_INDICATOR_RADIUS)
        if self._snap_indicator_item:
            try:
                self.canvas.coords(self._snap_indicator_item, *bbox)
                self.canvas.tag_raise(self._snap_indicator_item)
                return
            except tk.TclError:
                self._snap_indicator_item = None
        self._snap_indicator_item = self.canvas.create_oval(
            *bbox, outline=SNAP_INDICATOR_COLOR, fill="", width=2, dash=(3, 3))

    def _hide_snap_indicator(self):
        if self._snap_indicator_item:
            try:
                self.canvas.delete(self._snap_indicator_item)
            except tk.TclError:
                pass
            self._snap_indicator_item = None

    def _maybe_snap(self, ix, iy, exclude=None):
        """The nearest visible polygon vertex within SNAP_RADIUS screen px, else the point itself.

        Vertices only, never a point along an edge, so a shared boundary reuses
        the neighbour's own vertices. exclude is an (annotation id, vertex index)
        to skip, the vertex being dragged.
        """
        a = self.app
        if not a.snap_enabled:
            return (ix, iy)
        self._ensure_poly_bboxes()
        cx, cy = self.image_to_canvas(ix, iy)
        img_thr = SNAP_RADIUS / self.scale if self.scale > 0 else 1e9
        best_dist = SNAP_RADIUS
        best_pt = None
        for ann in self.visible_annotations():
            if ann.kind != "polygon":
                continue
            bbox = a._poly_bboxes.get(ann.id)
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                if (ix + img_thr < bx1 or ix - img_thr > bx2
                        or iy + img_thr < by1 or iy - img_thr > by2):
                    continue
            for vidx, (px, py) in enumerate(ann.points):
                if exclude is not None and (ann.id, vidx) == exclude:
                    continue
                pcx, pcy = self.image_to_canvas(px, py)
                dist = math.hypot(cx - pcx, cy - pcy)
                if dist < best_dist:
                    best_dist = dist
                    best_pt = (px, py)
        if best_pt:
            return best_pt
        return (ix, iy)

    # ──────────────────────────────────────────────────────────────────────────
    #  Mouse event dispatch
    # ──────────────────────────────────────────────────────────────────────────
    def _in_legend(self, event):
        """True when a canvas point falls on the drawn legend chip or panel."""
        box = self._legend_bbox
        return box is not None and box[0] <= event.x <= box[2] and box[1] <= event.y <= box[3]

    def on_button_press(self, event):
        self._legend_press = self._in_legend(event)
        if self._legend_press:
            self._legend_open = not self._legend_open
            self.display_image()
            return
        if not self.app._editable():
            return
        if self.app.mode == "box":
            self._box_press(event)
        else:
            self._poly_press(event)

    def on_move_press(self, event):
        if self._legend_press:
            return
        if self.app.mode == "box":
            self._box_drag(event)
        else:
            self._poly_drag(event)

    def on_button_release(self, event):
        if self._legend_press:
            self._legend_press = False
            return
        if self.app.mode == "box":
            self._box_release(event)
        else:
            self._poly_release(event)

    def _on_double_click(self, event):
        a = self.app
        if a.mode != "polygon" or self._in_legend(event):
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

    def _delete_annotation(self, ann_id):
        """Delete an annotation, dropping its item's verdict and resolving an open flag keyed by it.

        The verdict goes because the item it judged is gone: a miss has no key
        left, and a match's prediction is an unreviewed FP again. The flag has
        nothing left to open from.
        """
        a = self.app
        img_name = a.images[a.index]
        item = self._unfiltered_item_for(ann_id)
        self.engine.delete_annotation(ann_id)
        if item is not None:
            a._review.remove_verdict(img_name, item.key)
        a._review.resolve_flag(img_name, ann_id, a._current_user, note="deleted")
        a._rebuild_filter()
        a._update_filter_label()

    def on_right_click(self, event):
        a = self.app
        if self._in_legend(event) or not a._editable():
            return
        if a.mode == "polygon" and a.current_polygon:
            a.current_polygon = []
            a._vertex_redo_stack.clear()
            a._stream_active = False
            a._last_stream_pos = None
            self.display_image()
            return

        click_ix, click_iy = self.canvas_to_image(event.x, event.y)

        sel_id = a._selected_annotation_id
        if a.mode == "polygon" and sel_id is not None:
            if self._alive(sel_id):
                selected = a.document.get(sel_id)
                vertex_hit = self._find_nearest_vertex(
                    event.x, event.y, threshold=10)
                if vertex_hit and vertex_hit[0] == sel_id:
                    vi = vertex_hit[1]
                    self._push_undo()
                    if len(selected.points) <= 3:
                        self._delete_annotation(sel_id)
                    else:
                        new_pts = list(selected.points)
                        new_pts.pop(vi)
                        self.engine.set_points(sel_id, new_pts)
                    self._clear_drag_state()
                    a._mark_image_annotated()
                    self._refresh_review_after_edit(sel_id, a.images[a.index])
                    self.display_image()
                    a.update_title()
                    return
                if self._point_in_polygon(click_ix, click_iy,
                                          selected.points):
                    self._push_undo()
                    self._delete_annotation(sel_id)
                    self._clear_drag_state()
                    a._mark_image_annotated()
                    a._review_panel.refresh(keep_focus=True)
                    self.display_image()
                    a.update_title()
                    return
            a._selected_annotation_id = None
            self.display_image()
            return

        outlined = self._box_at_outline(event.x, event.y)
        if outlined is not None:
            self._push_undo()
            self._delete_annotation(outlined.id)
            a._mark_image_annotated()
            a._review_panel.refresh(keep_focus=True)
            self.display_image()
            a.update_title()
            return

        for ann in self.visible_annotations():
            if ann.kind != "polygon":
                continue
            if self._point_in_polygon(click_ix, click_iy, ann.points):
                self._push_undo()
                self._clear_drag_state()
                self._delete_annotation(ann.id)
                a._mark_image_annotated()
                a._review_panel.refresh(keep_focus=True)
                self.display_image()
                a.update_title()
                return

    # ──────────────────────────────────────────────────────────────────────────
    #  Box mode
    # ──────────────────────────────────────────────────────────────────────────
    def _unfiltered_item_for(self, ann_id):
        """The queue item holding that annotation under no filters, or None."""
        a = self.app
        if a.document is None or a.predictions_blind or not a.matches:
            return None
        items = build_queue(a.document, a.predictions, a.matches, a.verdicts)
        return next((qi for qi in items
                     if qi.annotation is not None and qi.annotation.id == ann_id), None)

    def _refresh_review_after_edit(self, ann_id, img_name):
        """Refresh the review queue after a GT geometry edit, moving its verdict with it.

        When the edit reclassifies the item under a new key (a match that
        became a miss, say), the verdict and flag history move to the new key,
        so the reviewer is not asked to re-confirm geometry they just fixed and
        the prediction left behind is unreviewed again. Focus follows the
        edited annotation when its new item is in the visible queue.
        """
        a = self.app
        old_item = self._unfiltered_item_for(ann_id)
        a._review_panel.refresh(keep_focus=True)
        if old_item is None:
            return
        new_item = self._unfiltered_item_for(ann_id)
        if new_item is None or new_item.key == old_item.key:
            return
        a._review.carry_flags(img_name, old_item.key, new_item.key)
        old_verdict = a.verdicts.get(old_item.key)
        if old_verdict is not None:
            if new_item.key not in a.verdicts:
                a._review.carry_verdict(img_name, new_item, old_verdict)
            a._review.remove_verdict(img_name, old_item.key)
        a._review_panel.refresh(keep_focus=True)
        for i, qi in enumerate(a.queue):
            if qi.key == new_item.key:
                a.queue_index = i
                break
        a._review_panel.update_labels()
        self.display_image()

    def _clear_box_edit_state(self):
        a = self.app
        a._box_edit_mode = None
        a._box_edit_anchor = None
        a._box_edit_origin = None
        a._box_edit_dirty = False

    def _box_edit_hit(self, ann, cx, cy):
        """Nearest-corner hit within tolerance (returns the fixed opposite corner),
        else "move" if inside the box body, else None."""
        (x1, y1), (x2, y2) = ann.points
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        opposite = [(x2, y2), (x1, y2), (x1, y1), (x2, y1)]
        best_i, best_d = None, BOX_CORNER_HIT_RADIUS
        for i, (px, py) in enumerate(corners):
            pcx, pcy = self.image_to_canvas(px, py)
            d = math.hypot(cx - pcx, cy - pcy)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None:
            return opposite[best_i]
        if self._box_outline_distance(ann, cx, cy) <= BOX_EDGE_HIT_RADIUS:
            return "move"
        return None

    def _box_outline_distance(self, ann, cx, cy):
        """Canvas-pixel distance from (cx, cy) to the nearest edge of a box annotation."""
        (x1, y1), (x2, y2) = ann.points
        corners = [self.image_to_canvas(x, y) for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        return min(point_to_segment_dist(cx, cy, *corners[i], *corners[(i + 1) % 4])
                   for i in range(4))

    def _box_at_outline(self, cx, cy):
        """The visible box whose outline is nearest (cx, cy) within BOX_EDGE_HIT_RADIUS, or None.

        Boxes are picked by their outline, never their interior, so a press inside
        a box can start a new box, e.g. a bur sitting inside a neighbour's box.
        """
        best, best_d = None, BOX_EDGE_HIT_RADIUS
        for ann in self.visible_annotations():
            if ann.kind != "box":
                continue
            d = self._box_outline_distance(ann, cx, cy)
            if d <= best_d:
                best, best_d = ann, d
        return best

    def _start_box_move(self, ann, event):
        """Begin dragging a box by its outline; a release without movement changes nothing."""
        a = self.app
        p1, p2 = ann.points
        ix, iy = self.canvas_to_image(event.x, event.y)
        a._box_edit_mode = "move"
        a._box_edit_origin = (p1, p2, ix, iy)
        a._box_edit_dirty = False
        self.canvas.config(cursor="fleur")

    def _box_press(self, event):
        a = self.app
        # Dropped here so a press that starts no rectangle cannot commit one on release.
        a.start_x = None
        a.start_y = None
        sel_id = a._selected_annotation_id
        if sel_id is not None and a._annotation_visible and self._alive(sel_id):
            selected = a.document.get(sel_id)
            if selected.kind == "box":
                hit = self._box_edit_hit(selected, event.x, event.y)
                if hit == "move":
                    self._start_box_move(selected, event)
                    return
                if hit is not None:
                    a._box_edit_mode = "resize"
                    a._box_edit_anchor = hit
                    a._box_edit_dirty = False
                    self.canvas.config(cursor="fleur")
                    return
        outlined = self._box_at_outline(event.x, event.y)
        if outlined is not None:
            self.select_annotation(outlined.id)
            self._start_box_move(outlined, event)
            return
        if a._selected_annotation_id is not None:
            self.select_annotation(None)
        if a._review_filter_class == "all":
            a.show_banner("Select a class before drawing.")
            return
        a.start_x = event.x
        a.start_y = event.y
        color = a._get_class_color(a.active_class)
        a.rect = self.canvas.create_rectangle(
            a.start_x, a.start_y, a.start_x, a.start_y,
            outline=color, width=2)

    def _box_drag(self, event):
        a = self.app
        sel_id = a._selected_annotation_id
        if a._box_edit_mode == "resize":
            if not self._alive(sel_id):
                self._clear_box_edit_state()
                return
            ax, ay = a._box_edit_anchor
            ix, iy = self.canvas_to_image(event.x, event.y)
            ix = max(0, min(a.img_width, ix))
            iy = max(0, min(a.img_height, iy))
            if not a._box_edit_dirty:
                self._push_undo()
                a._box_edit_dirty = True
            self.engine.set_points(
                sel_id, ((min(ax, ix), min(ay, iy)), (max(ax, ix), max(ay, iy))))
            self.display_image()
            return
        if a._box_edit_mode == "move":
            if not self._alive(sel_id):
                self._clear_box_edit_state()
                return
            (ox1, oy1), (ox2, oy2), start_ix, start_iy = a._box_edit_origin
            ix, iy = self.canvas_to_image(event.x, event.y)
            dx = _clamp_delta(ix - start_ix, -ox1, a.img_width - ox2)
            dy = _clamp_delta(iy - start_iy, -oy1, a.img_height - oy2)
            if not a._box_edit_dirty:
                self._push_undo()
                a._box_edit_dirty = True
            self.engine.set_points(
                sel_id, ((ox1 + dx, oy1 + dy), (ox2 + dx, oy2 + dy)))
            self.display_image()
            return
        if (a.rect and a.start_x is not None
                and a.start_y is not None):
            self.canvas.coords(
                a.rect, a.start_x, a.start_y, event.x, event.y)

    def _box_release(self, event):
        a = self.app
        if a._box_edit_mode is not None:
            sel_id = a._selected_annotation_id
            mode = a._box_edit_mode
            dirty = a._box_edit_dirty
            self._clear_box_edit_state()
            self.canvas.config(cursor="cross")
            if dirty and self._alive(sel_id):
                (x1, y1), (x2, y2) = a.document.get(sel_id).points
                if mode == "resize" and (x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE):
                    # Roll the degenerate box back, then drop the snapshot undo_snapshot
                    # pushed onto the redo stack so Ctrl+Y cannot restore it.
                    if self.engine.undo_snapshot():
                        a._redo_stack.pop()
                else:
                    a._mark_image_annotated()
                    self._refresh_review_after_edit(sel_id, a.images[a.index])
            self.display_image()
            a.update_title()
            return
        if a.start_x is None or a.start_y is None:
            return
        ix1, iy1 = self.canvas_to_image(a.start_x, a.start_y)
        ix2, iy2 = self.canvas_to_image(event.x, event.y)

        x1 = max(0, min(ix1, ix2))
        y1 = max(0, min(iy1, iy2))
        x2 = min(a.img_width, max(ix1, ix2))
        y2 = min(a.img_height, max(iy1, iy2))

        if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
            if a.rect:
                self.canvas.delete(a.rect)
            a.rect = None
            return

        self._push_undo()
        added = self.engine.add_box(x1, y1, x2, y2)
        a._mark_image_annotated()
        a._record_annotation_added()
        a.rect = None
        a.accept_drawn_annotation(added)
        a.update_title()

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

        sel_id = a._selected_annotation_id
        if sel_id is not None:
            if self._alive(sel_id) and a.document.get(sel_id).kind == "polygon":
                pts_sel = a.document.get(sel_id).points
                best_vi, best_vd = None, 8
                for vi, (px, py) in enumerate(pts_sel):
                    vcx, vcy = self.image_to_canvas(px, py)
                    d = math.hypot(event.x - vcx, event.y - vcy)
                    if d < best_vd:
                        best_vd = d
                        best_vi = vi
                if best_vi is not None:
                    # Undo is pushed once the pointer actually moves; a release
                    # without movement starts a new polygon on this vertex instead.
                    a._dragging_vertex = (sel_id, best_vi)
                    a._drag_orig_pos = pts_sel[best_vi]
                    self._vertex_press = (event.x, event.y)
                    self._vertex_drag_started = False
                    return
                best_ei, best_ed, best_ept = None, 6, None
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
                    new_points = list(pts_sel)
                    new_points.insert(best_ei + 1, best_ept)
                    self.engine.set_points(sel_id, new_points)
                    a._dragging_vertex = (sel_id, best_ei + 1)
                    a._drag_orig_pos = best_ept
                    self.canvas.config(cursor="fleur")
                    self.display_image()
                    return
            just_deselected = True
            a._selected_annotation_id = None
        else:
            just_deselected = False

        # With snapping on, a click that snaps to a vertex means "start here", not "select".
        snaps_to_vertex = self._maybe_snap(ix, iy) != (ix, iy)
        if not snaps_to_vertex:
            if not a.snap_enabled:
                vhit = self._find_nearest_vertex(event.x, event.y, threshold=15)
                if vhit:
                    a._selected_annotation_id = vhit[0]
                    self.display_image()
                    return
            for ann in self.visible_annotations():
                if ann.kind != "polygon":
                    continue
                if self._point_in_polygon(ix, iy, ann.points):
                    a._selected_annotation_id = ann.id
                    self.display_image()
                    return

            if just_deselected:
                self.display_image()
                return

        self._start_polygon(ix, iy)

    def _start_polygon(self, ix, iy):
        """Start a new polygon at an image point, snapped, and begin streaming if Stream is on."""
        a = self.app
        if a._review_filter_class == "all":
            a.show_banner("Select a class before drawing.")
            return
        ix, iy = self._clamp(*self._maybe_snap(ix, iy))
        a.current_polygon = [(ix, iy)]
        a._vertex_redo_stack.clear()
        if a._stream_mode:
            a._stream_active = True
            a._last_stream_pos = (ix, iy)
        self.display_image()

    def _poly_drag(self, event):
        a = self.app
        if a._dragging_vertex is None:
            return
        ann_id, vi = a._dragging_vertex
        if not self._alive(ann_id) or ann_id != a._selected_annotation_id:
            self._clear_drag_state()
            self._vertex_press = None
            return
        if not self._vertex_drag_started and self._vertex_press is not None:
            px, py = self._vertex_press
            if math.hypot(event.x - px, event.y - py) < CLICK_SLOP:
                return
            self._push_undo()
            self._vertex_drag_started = True
            self.canvas.config(cursor="fleur")
        raw_ix, raw_iy = self.canvas_to_image(event.x, event.y)
        ix, iy = self._maybe_snap(raw_ix, raw_iy, exclude=(ann_id, vi))
        did_snap = (ix, iy) != (raw_ix, raw_iy)
        ix = max(0, min(a.img_width, ix))
        iy = max(0, min(a.img_height, iy))
        new_points = list(a.document.get(ann_id).points)
        new_points[vi] = (ix, iy)
        self.engine.set_points(ann_id, new_points)
        self.display_image()
        if a.snap_enabled:
            if did_snap:
                self._show_snap_indicator(*self.image_to_canvas(ix, iy))
            else:
                self._hide_snap_indicator()

    def _poly_release(self, event):
        a = self.app
        if a._dragging_vertex is not None and self._vertex_press is not None \
                and not self._vertex_drag_started:
            ann_id, vi = a._dragging_vertex
            self._vertex_press = None
            self._clear_drag_state()
            if self._alive(ann_id):
                vx, vy = a.document.get(ann_id).points[vi]
                a._selected_annotation_id = None
                self._start_polygon(vx, vy)
            return
        self._vertex_press = None
        self._vertex_drag_started = False
        if a._dragging_vertex is not None:
            ann_id = a._dragging_vertex[0]
            a._mark_image_annotated()
            a._dragging_vertex = None
            a._drag_orig_pos = None
            self.canvas.config(cursor="cross")
            self._refresh_review_after_edit(ann_id, a.images[a.index])

    def _close_polygon(self):
        a = self.app
        added = self.engine.close_current_polygon()
        if not added:
            self.display_image()
            return
        a._mark_image_annotated()
        a._record_annotation_added()
        self._poly_preview_line = None
        a.accept_drawn_annotation(added)
        a.update_title()

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
        for ann in self.visible_annotations():
            if ann.kind != "polygon":
                continue
            bbox = a._poly_bboxes.get(ann.id)
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                if (qix + img_thr < bx1 or qix - img_thr > bx2
                        or qiy + img_thr < by1 or qiy - img_thr > by2):
                    continue
            for vi, (px, py) in enumerate(ann.points):
                vcx, vcy = self.image_to_canvas(px, py)
                dist = math.hypot(cx - vcx, cy - vcy)
                if dist < best_dist:
                    best_dist = dist
                    best = (ann.id, vi)
        return best

    def _find_nearest_edge_selected(self, cx, cy, threshold=6):
        a = self.app
        ann_id = a._selected_annotation_id
        if not self._alive(ann_id):
            return None
        ann = a.document.get(ann_id)
        if ann.kind != "polygon":
            return None
        points = ann.points
        n = len(points)
        for ei in range(n):
            ax, ay = self.image_to_canvas(*points[ei])
            bx, by = self.image_to_canvas(*points[(ei + 1) % n])
            dist = point_to_segment_dist(cx, cy, ax, ay, bx, by)
            if dist < threshold:
                return ann_id
        return None

    _point_in_polygon = staticmethod(point_in_polygon)

    # ──────────────────────────────────────────────────────────────────────────
    #  Navigation
    # ──────────────────────────────────────────────────────────────────────────
    def next_index(self):
        """The index a step forward lands on, honouring the active image filter."""
        a = self.app
        if a._active_filter != "all" and a._filtered_indices:
            for idx in a._filtered_indices:
                if idx > a.index:
                    return idx
            return a._filtered_indices[0]
        return a.index + 1

    def next_image(self, event=None):
        """Step to the next image, asking first when this one is not marked complete."""
        if self.app.confirm_leaving_incomplete():
            self.app.go_to_image(self.next_index())

    def prev_image(self, event=None):
        a = self.app
        if a._active_filter != "all" and a._filtered_indices:
            for idx in reversed(a._filtered_indices):
                if idx < a.index:
                    a.go_to_image(idx)
                    return
            a.go_to_image(a._filtered_indices[-1])
            return
        a.go_to_image(a.index - 1)

    # ──────────────────────────────────────────────────────────────────────────
    #  Undo / Redo
    # ──────────────────────────────────────────────────────────────────────────
    def _push_undo(self):
        self.engine.push_undo()

    def undo_last(self, event=None):
        """Take back the last in-progress vertex, or else the last document change."""
        a = self.app
        if a.current_polygon:
            a._vertex_redo_stack.append(a.current_polygon.pop())
            self.display_image()
            return
        if self.engine.undo_snapshot():
            self.canvas.config(cursor="cross")
            self.display_image()
            a.update_title()

    def redo_last(self, event=None):
        """Put back the last undone in-progress vertex, or else the last undone document change."""
        a = self.app
        if a._vertex_redo_stack:
            a.current_polygon.append(a._vertex_redo_stack.pop())
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
        """Save the current document. Returns None or the engine's error message."""
        a = self.app
        if a.document is not None:
            print(f"[YoloLabeler] Saving annotations for {a.images[a.index]} "
                  f"({len(a.document.annotations)} annotations)")
        return self.engine.save()

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
        sel_vert_r = max(vert_r + 2, 7,
                         min(VERTEX_HANDLE_RADIUS * (2.2 - s * 0.2), 16))
        label_size = max(7, min(int(9 * (0.6 + s * 0.4)), 18))
        dash_a = max(2, int(4 * (0.5 + s * 0.5)))
        dash_b = max(2, int(4 * (0.5 + s * 0.5)))
        placed_labels = []

        def _halo(x, y, text, fill, **kw):
            place_label(canvas, placed_labels, x, y, text, fill, **kw)

        drag_canvas_pt = None
        if a._dragging_vertex is not None and a.snap_enabled:
            drag_id, dvi = a._dragging_vertex
            if self._alive(drag_id):
                dpts = a.document.get(drag_id).points
                if dvi < len(dpts):
                    drag_canvas_pt = self.image_to_canvas(*dpts[dvi])

        focus_ann = a.queue[a.queue_index].annotation if (
            a.queue and 0 <= a.queue_index < len(a.queue)) else None
        gold_ann_id = (focus_ann.id if focus_ann is not None and a._annotation_visible
                       else None)

        # Predictions go down first so a solid annotation and its selection handles sit on top.
        draw_prediction_layer(
            self.canvas, self.image_to_canvas, a, a.class_names, a.font_family,
            label_size, show_gt=a._annotation_visible, show_pred=a._review_show_pred,
            class_color=a._get_class_color, placed_labels=placed_labels,
            line_w=line_w)
        statuses = status_colors_active(a, a._review_show_pred)

        for ann in self.visible_annotations():
            class_id = ann.class_id
            is_selected = (ann.id == a._selected_annotation_id)
            if ann.id == gold_ann_id and not is_selected:
                continue
            if statuses is not None:
                color = status_color(statuses, ann.id)
            else:
                color = a._get_class_color(class_id)
            class_name = a.class_names.get(class_id, str(class_id))
            if ann.kind == "box":
                (x1, y1), (x2, y2) = ann.points
                if x2 < vis_x1 or x1 > vis_x2 or y2 < vis_y1 or y1 > vis_y2:
                    continue
                cx1, cy1 = self.image_to_canvas(x1, y1)
                cx2, cy2 = self.image_to_canvas(x2, y2)
                canvas.create_rectangle(
                    cx1, cy1, cx2, cy2, outline=SELECTION_COLOR if is_selected else color,
                    width=line_w + 1 if is_selected else line_w)
                if is_selected:
                    r = sel_vert_r
                    for hx, hy in ((cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)):
                        canvas.create_rectangle(
                            hx - r, hy - r, hx + r, hy + r,
                            fill="white", outline=SELECTION_COLOR, width=2)
                _halo(cx1 + 2, cy1 - 2, anchor="sw",
                      text=f"{class_id}: {class_name}",
                      fill=color,
                      font=(a.font_family, label_size, "bold"))
                continue

            points = ann.points
            if points and not is_selected:
                pxs = [p[0] for p in points]
                pys = [p[1] for p in points]
                if (max(pxs) < vis_x1 or min(pxs) > vis_x2
                        or max(pys) < vis_y1 or min(pys) > vis_y2):
                    continue
            canvas_pts = []
            for px, py in points:
                cx, cy = self.image_to_canvas(px, py)
                canvas_pts.extend([cx, cy])
            if len(canvas_pts) >= 6:
                canvas.create_polygon(
                    *canvas_pts, outline=SELECTION_COLOR if is_selected else color,
                    fill="", width=poly_w + 1 if is_selected else poly_w)
            show_verts = (
                is_selected
                or ann.id == a._hovered_annotation_id
                or (a._dragging_vertex is not None
                    and a._dragging_vertex[0] == ann.id)
            )
            if not show_verts and drag_canvas_pt is not None and points:
                dcx, dcy = drag_canvas_pt
                for px, py in points:
                    pcx, pcy = self.image_to_canvas(px, py)
                    if math.hypot(dcx - pcx, dcy - pcy) < SNAP_RADIUS * 3:
                        show_verts = True
                        break
            if show_verts:
                r = sel_vert_r if is_selected else vert_r
                fill = SELECTION_COLOR if is_selected else color
                for px, py in points:
                    cx, cy = self.image_to_canvas(px, py)
                    canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        fill=fill, outline="white", width=2 if is_selected else 1)
            if points:
                lx, ly = self.image_to_canvas(*points[0])
                _halo(lx + 2, ly - 2, anchor="sw",
                      text=f"{class_id}: {class_name}",
                      fill=color,
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

        help_y0 = 10
        if a.banner_text:
            banner_h = self._draw_block(a.banner_text.split("\n"), y0=10)
            help_y0 = 10 + banner_h + 10
        self.render_help(help_y0)
        self.render_legend()
        self._update_snap_indicator()

    def _legend_classes(self):
        """Class ids drawn on this image, from its annotations and predictions."""
        a = self.app
        ids = {ann.class_id for ann in a.document.annotations} if a.document else set()
        if a._review_show_pred and not a.predictions_blind:
            ids |= {p.class_id for p in a.predictions if p.confidence >= a.conf_threshold}
        return sorted(ids)

    def render_legend(self):
        """Draw the symbology legend in the lower left: a chip, or the open panel above it."""
        a = self.app
        canvas = self.canvas
        ch = canvas.winfo_height() or 800
        font = (a.font_family, 12)
        fnt = tkFont.Font(family=a.font_family, size=12)
        line_h = fnt.metrics("linespace") + 8
        pad, swatch_w, x0 = 10, 40, 10
        chip = "Legend ▾" if self._legend_open else "Legend ▴"
        chip_w = fnt.measure(chip) + pad * 2
        chip_y1 = ch - 10
        chip_y0 = chip_y1 - line_h
        canvas.create_rectangle(x0, chip_y0, x0 + chip_w, chip_y1, fill=LEGEND_BG,
                                outline=LEGEND_BORDER, width=1, tags="legend")
        canvas.create_text(x0 + pad, (chip_y0 + chip_y1) / 2, anchor="w", text=chip,
                           fill=FG_COLOR, font=font, tags="legend")
        if not self._legend_open:
            self._legend_bbox = (x0, chip_y0, x0 + chip_w, chip_y1)
            return

        style = LayerStyle()
        if status_colors_active(a, a._review_show_pred) is not None:
            rows = [(("class", STATUS_COLORS["accepted"]), "Green: accepted"),
                    (("class", STATUS_COLORS["not_reviewed"]), "Yellow: not reviewed"),
                    (("class", STATUS_COLORS["rejected"]), "Red: rejected")]
        else:
            rows = [(("class", a._get_class_color(cid)),
                     f"{cid}: {a.class_names.get(cid, cid)}") for cid in self._legend_classes()]
        rows += [
            (("line", FG_COLOR, None), "Solid: annotation (ground truth)"),
            (("line", FG_COLOR, style.dash), "Dashed: prediction"),
            (("line", FG_COLOR, style.rejected_dash), "Dotted: rejected prediction"),
            (("halo",), "Blue glow: item in review focus"),
            (("flag",), f"White {FLAG_MARK}: flagged for a second look (c)"),
            (("selected",), "Blue with handles: selected for editing"),
            (("snap",), "Dashed ring: snap target"),
        ]
        text_w = max(fnt.measure(text) for _, text in rows)
        panel_w = pad * 3 + swatch_w + text_w
        panel_y1 = chip_y0 - 4
        panel_y0 = panel_y1 - pad * 2 - line_h * len(rows)
        canvas.create_rectangle(x0, panel_y0, x0 + panel_w, panel_y1, fill=LEGEND_BG,
                                outline=LEGEND_BORDER, width=1, tags="legend")
        for i, (swatch, text) in enumerate(rows):
            cy = panel_y0 + pad + line_h * i + line_h / 2
            sx0, sx1 = x0 + pad, x0 + pad + swatch_w
            kind = swatch[0]
            if kind == "class":
                canvas.create_rectangle(sx0, cy - 5, sx1, cy + 5, outline=swatch[1],
                                        width=3, fill="", tags="legend")
            elif kind == "line":
                canvas.create_line(sx0, cy, sx1, cy, fill=swatch[1], width=3,
                                   dash=swatch[2] or "", tags="legend")
            elif kind == "halo":
                canvas.create_line(sx0, cy, sx1, cy, fill=SELECTION_COLOR, width=7,
                                   tags="legend")
                canvas.create_line(sx0, cy, sx1, cy, fill=FG_COLOR, width=2, tags="legend")
            elif kind == "flag":
                canvas.create_text((sx0 + sx1) / 2, cy, text=FLAG_MARK, fill=FLAG_COLOR,
                                   font=(a.font_family, 14, "bold"), tags="legend")
            elif kind == "selected":
                canvas.create_line(sx0, cy, sx1, cy, fill=SELECTION_COLOR, width=3,
                                   tags="legend")
                for hx in (sx0 + 3, sx1 - 3):
                    canvas.create_oval(hx - 4, cy - 4, hx + 4, cy + 4, fill=SELECTION_COLOR,
                                       outline="white", width=2, tags="legend")
            else:
                mid = (sx0 + sx1) / 2
                canvas.create_oval(mid - SNAP_INDICATOR_RADIUS, cy - SNAP_INDICATOR_RADIUS,
                                   mid + SNAP_INDICATOR_RADIUS, cy + SNAP_INDICATOR_RADIUS,
                                   outline=SNAP_INDICATOR_COLOR, width=2, dash=(3, 3),
                                   tags="legend")
            canvas.create_text(sx1 + pad, cy, anchor="w", text=text, fill=FG_COLOR,
                               font=font, tags="legend")
        self._legend_bbox = (x0, panel_y0, x0 + max(panel_w, chip_w), chip_y1)

    def _draw_block(self, lines, y0):
        """Draw a padded text block at x 10, y0, shared by the banner and the help overlay; returns its height."""
        canvas = self.canvas
        font_family = "Menlo" if sys.platform == "darwin" else "Consolas"
        font_size = 14
        pad = 14

        fnt = tkFont.Font(family=font_family, size=font_size)
        line_height = fnt.metrics("linespace") + 2
        max_text_w = max(fnt.measure(ln) for ln in lines) if lines else 100

        block_w = max_text_w + pad * 3
        block_h = len(lines) * line_height + pad * 2
        x0 = 10

        canvas.create_rectangle(
            x0, y0, x0 + block_w, y0 + block_h,
            fill="#1A1A1A", outline="#444444", width=1, stipple="")

        for i, line in enumerate(lines):
            canvas.create_text(
                x0 + pad, y0 + pad + i * line_height,
                anchor="nw", text=line,
                fill=FG_COLOR, font=(font_family, font_size))
        return block_h

    def render_help(self, y0=10):
        a = self.app
        if not a.show_help:
            return

        item = a._review_panel.current_item()
        help_lines = keybindings.help_lines(
            a.mode, has_queue=bool(a.queue), has_pair=bool(item and item.annotation))

        self._draw_block(help_lines, y0)
