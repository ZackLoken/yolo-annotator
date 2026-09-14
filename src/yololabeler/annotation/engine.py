"""AnnotationEngine: headless CRUD, undo/redo and save over the current Document."""

import os

from yololabeler.annotation.document import new_annotation, save_document
from yololabeler.state import AppState

# Snapshot count kept for undo; value carried over from the original implementation.
UNDO_DEPTH = 30


class AnnotationEngine:
    """Annotation logic that operates on AppState without any GUI dependency."""

    def __init__(self, state: AppState):
        self.state = state

    # ── Spatial index ──────────────────────────────────────────────────────

    def invalidate_poly_bboxes(self):
        """Mark the polygon bounding-box cache stale."""
        self.state._poly_bboxes_dirty = True

    def ensure_poly_bboxes(self):
        """Rebuild the polygon bounding-box cache, keyed by annotation id, if stale."""
        s = self.state
        if not s._poly_bboxes_dirty:
            return
        s._poly_bboxes = {}
        if s.document is not None:
            for a in s.document.polygons():
                xs = [p[0] for p in a.points]
                ys = [p[1] for p in a.points]
                s._poly_bboxes[a.id] = (min(xs), min(ys), max(xs), max(ys))
        s._poly_bboxes_dirty = False

    # ── Undo / redo ────────────────────────────────────────────────────────

    def _snapshot(self):
        s = self.state
        return (s.document.snapshot(), s._selected_annotation_id, dict(s.verdicts))

    def _restore(self, snap):
        s = self.state
        annotations, selected, verdicts = snap
        s.document.restore(annotations)
        s._selected_annotation_id = selected
        s.verdicts.clear()
        s.verdicts.update(verdicts)
        self.invalidate_poly_bboxes()
        self.clear_drag_state()

    def push_undo(self):
        """Snapshot the document, selection and verdicts before a mutation."""
        s = self.state
        s._undo_stack.append(self._snapshot())
        s._redo_stack.clear()
        if len(s._undo_stack) > UNDO_DEPTH:
            s._undo_stack.pop(0)

    def undo_snapshot(self):
        """Restore the previous snapshot; returns False when there is none."""
        s = self.state
        if not s._undo_stack:
            return False
        s._redo_stack.append(self._snapshot())
        self._restore(s._undo_stack.pop())
        return True

    def redo_snapshot(self):
        """Re-apply the last undone snapshot; returns False when there is none."""
        s = self.state
        if not s._redo_stack:
            return False
        s._undo_stack.append(self._snapshot())
        self._restore(s._redo_stack.pop())
        return True

    def clear_drag_state(self):
        """Reset in-progress vertex-drag and hover state."""
        s = self.state
        s._dragging_vertex = None
        s._drag_orig_pos = None
        s._hovered_annotation_id = None

    # ── CRUD ───────────────────────────────────────────────────────────────

    def add_box(self, x1, y1, x2, y2):
        """Append a drawn box for the active class; caller pushes undo first."""
        s = self.state
        a = new_annotation("box", ((x1, y1), (x2, y2)), s.active_class, s._current_user)
        s.document.add(a)
        return a

    def close_current_polygon(self):
        """Finalize the in-progress polygon, clamped to image bounds; None if under 3 vertices."""
        s = self.state
        if len(s.current_polygon) < 3:
            s.current_polygon = []
            return None
        clamped = [(max(0, min(s.img_width, x)), max(0, min(s.img_height, y)))
                   for x, y in s.current_polygon]
        self.push_undo()
        a = new_annotation("polygon", clamped, s.active_class, s._current_user)
        s.document.add(a)
        self.invalidate_poly_bboxes()
        s.current_polygon = []
        return a

    def delete_annotation(self, ann_id):
        """Remove and return the annotation with the given id."""
        removed = self.state.document.remove(ann_id)
        self.invalidate_poly_bboxes()
        if self.state._selected_annotation_id == ann_id:
            self.state._selected_annotation_id = None
        return removed

    def set_points(self, ann_id, points):
        """Replace the points of the given annotation in place."""
        self.state.document.replace(ann_id, points=points)
        self.invalidate_poly_bboxes()

    # ── I/O ────────────────────────────────────────────────────────────────

    def label_paths(self):
        """Return the (detect, segment, sidecar) file paths for the current image."""
        s = self.state
        stem = os.path.splitext(s.images[s.index])[0]
        return (os.path.join(s.detect_dir, f"{stem}.txt"),
                os.path.join(s.segment_dir, f"{stem}.txt"),
                os.path.join(s.state_dir, "annotations", f"{stem}.json"))

    def save(self):
        """Write labels and sidecar for the current image; returns None or an error message."""
        s = self.state
        if s.document is None or not s.images:
            return "No image loaded"
        if s.img_width <= 0 or s.img_height <= 0:
            return f"Invalid image size {s.img_width}x{s.img_height}"
        os.makedirs(s.detect_dir, exist_ok=True)
        os.makedirs(s.segment_dir, exist_ok=True)
        detect, segment, sidecar = self.label_paths()
        try:
            save_document(s.document, detect, segment, sidecar)
        except OSError as e:
            target = e.filename2 or e.filename or detect
            return f"Could not save {target}: {e.strerror or e}"
        return None
