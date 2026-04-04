"""AnnotationEngine — Annotation CRUD, spatial index, undo/redo.

GUI-free.  Operates on an AppState instance.  Can be instantiated
headlessly for programmatic use (AI agents, training pipelines, CLI).
"""

import os

from yololabeler.state import AppState
from yololabeler.label_io import write_detect_labels, write_segment_labels


class AnnotationEngine:
    """Annotation logic that operates on AppState without any GUI dependency."""

    def __init__(self, state: AppState):
        self.state = state

    # ── Spatial index (polygon bounding-box cache) ────────────────────────

    def invalidate_poly_bboxes(self):
        self.state._poly_bboxes_dirty = True

    def ensure_poly_bboxes(self):
        s = self.state
        if not s._poly_bboxes_dirty:
            return
        s._poly_bboxes = []
        for points, _ in s.polygons:
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                s._poly_bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                s._poly_bboxes.append((0, 0, 0, 0))
        s._poly_bboxes_dirty = False

    # ── Undo / redo ───────────────────────────────────────────────────────

    def push_undo(self):
        """Snapshot current annotation state before a mutation."""
        s = self.state
        snapshot = (
            list(s.boxes),
            list(s.polygons),
            s._selected_polygon_idx,
        )
        s._undo_stack.append(snapshot)
        s._redo_stack.clear()
        if len(s._undo_stack) > 30:
            s._undo_stack.pop(0)

    def undo_snapshot(self):
        """Restore previous state from snapshot stack.

        Returns True if state was restored, False if stack was empty.
        """
        s = self.state
        if not s._undo_stack:
            return False
        redo_snap = (
            list(s.boxes),
            list(s.polygons),
            s._selected_polygon_idx,
        )
        s._redo_stack.append(redo_snap)
        boxes, polygons, sel_idx = s._undo_stack.pop()
        s.boxes = boxes
        s.polygons = polygons
        self.invalidate_poly_bboxes()
        s._selected_polygon_idx = sel_idx
        self.clear_drag_state()
        return True

    def redo_snapshot(self):
        """Restore state from redo stack.

        Returns True if state was restored, False if stack was empty.
        """
        s = self.state
        if not s._redo_stack:
            return False
        undo_snap = (
            list(s.boxes),
            list(s.polygons),
            s._selected_polygon_idx,
        )
        s._undo_stack.append(undo_snap)
        boxes, polygons, sel_idx = s._redo_stack.pop()
        s.boxes = boxes
        s.polygons = polygons
        self.invalidate_poly_bboxes()
        s._selected_polygon_idx = sel_idx
        self.clear_drag_state()
        return True

    # ── Drag / selection helpers ──────────────────────────────────────────

    def clear_drag_state(self):
        s = self.state
        s._dragging_vertex = None
        s._drag_orig_pos = None
        s._hovered_polygon_idx = None

    # ── Polygon CRUD ──────────────────────────────────────────────────────

    def close_current_polygon(self):
        """Finalize the in-progress polygon.

        Clamps vertices to image bounds, pushes undo, appends polygon.
        Returns True if a polygon was added, False if < 3 vertices.
        """
        s = self.state
        if len(s.current_polygon) < 3:
            s.current_polygon = []
            return False
        clamped = []
        for x, y in s.current_polygon:
            clamped.append((
                max(0, min(s.img_width, x)),
                max(0, min(s.img_height, y)),
            ))
        self.push_undo()
        s.polygons.append((clamped, s.active_class))
        self.invalidate_poly_bboxes()
        s.current_polygon = []
        return True

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self):
        """Save current annotations to YOLO-format label files.

        Returns True on success, False if save was skipped or failed.
        """
        s = self.state
        if (not s.images or not s.labels_dir
                or not s.detect_dir or not s.segment_dir):
            return False
        os.makedirs(s.detect_dir, exist_ok=True)
        os.makedirs(s.segment_dir, exist_ok=True)
        stem = os.path.splitext(s.images[s.index])[0]
        detect_path = os.path.join(s.detect_dir, f"{stem}.txt")
        segment_path = os.path.join(s.segment_dir, f"{stem}.txt")
        if s.img_width <= 0 or s.img_height <= 0:
            print(f"Warning: Invalid image dimensions "
                  f"({s.img_width}x{s.img_height}), skipping save")
            return False
        try:
            write_detect_labels(detect_path, s.boxes,
                                s.img_width, s.img_height)
            write_segment_labels(segment_path, s.polygons,
                                 s.img_width, s.img_height)
            return True
        except OSError as e:
            print(f"Warning: Could not save annotations: {e}")
            return False
