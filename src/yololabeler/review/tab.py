"""ReviewTab — Review domain module for YoloLabeler.

Consolidates all review-tab functionality: canvas construction, bindings,
coordinate conversion, pan/zoom, image loading, detection management,
navigation, filter callbacks, review actions (accept/reject/edit),
advancement logic, and canvas rendering (absorbed from ReviewRenderer).
"""

import os
import sys
import time
import tkinter as tk
import tkinter.font as tkFont

from PIL import Image, ImageTk

import customtkinter as ctk

from yololabeler.label_io import parse_detect_labels, parse_segment_labels
from yololabeler.rendering import halo_text

# Constants (duplicated from annotator to avoid circular import)
FG_COLOR = "#E0E0E0"
CANVAS_BG = "#2D2D2D"
BG_COLOR = "#1E1E1E"
ACCENT_HOVER = "#608864"
SI_GREEN = "#507754"
SI_PERSIMMON = "#E6976B"
REVIEW_IOU_THRESHOLD = 0.60
REVIEW_CONF_THRESHOLD = 0.50


class ReviewTab:
    """Review tab domain module — owns all review-tab behaviour.

    Parameters
    ----------
    app : YoloLabeler
        The main application instance.  State attributes are accessed
        via ``self.app.X`` (transparently forwarded to AppState).
        GUI widgets and non-review methods also live on *app*.
    """

    def __init__(self, app):
        self.app = app
        self.engine = app._review

    # ══════════════════════════════════════════════════════════════════════════
    #  Construction
    # ══════════════════════════════════════════════════════════════════════════

    def build(self):
        """Create the review canvas and set up bindings."""
        a = self.app
        a._review_canvas = tk.Canvas(a._review_frame, bg=CANVAS_BG,
                                     highlightthickness=0)
        a._review_canvas.pack(fill="both", expand=True)
        self._setup_review_bindings()

    def _setup_review_bindings(self):
        """Canvas bindings for the review tab (zoom and pan only)."""
        c = self.app._review_canvas
        if sys.platform == "darwin":
            c.bind("<ButtonPress-3>", self._review_pan_start)
            c.bind("<B3-Motion>", self._review_pan_drag)
        else:
            c.bind("<ButtonPress-2>", self._review_pan_start)
            c.bind("<B2-Motion>", self._review_pan_drag)
        c.bind("<Control-MouseWheel>", self._review_zoom)
        c.bind("<MouseWheel>", self._review_scroll)
        c.bind("<Shift-MouseWheel>", self._review_hscroll)
        # Linux scroll bindings
        c.bind("<Button-4>", lambda e: self._review_scroll_linux(e, 1))
        c.bind("<Button-5>", lambda e: self._review_scroll_linux(e, -1))
        c.bind("<Control-Button-4>", lambda e: self._review_zoom_linux(e, 1))
        c.bind("<Control-Button-5>", lambda e: self._review_zoom_linux(e, -1))
        c.bind("<Shift-Button-4>", lambda e: self._review_hscroll_linux(e, 1))
        c.bind("<Shift-Button-5>", lambda e: self._review_hscroll_linux(e, -1))
        c.bind("<Configure>", self._on_review_canvas_configure)

    def activate(self):
        """Called when switching TO the Review tab."""
        a = self.app
        # Record annotate time, start review time
        a._record_image_time()
        if a._review_image_start_time is None:
            a._review_image_start_time = time.time()
        # Back up original labels before any review actions
        self.engine.backup_original_labels()
        # Clean up prediction reference from annotate
        if a._review_return_pending:
            a._annotate_pred_reference = None
            a._review_return_pending = False
        # Show review controls, hide annotate-only toolbar sections
        a._toolbar_center.pack_forget()
        a._review_status_frame.pack(side="left", fill="x", expand=True)
        # Show counts in the right-side status area
        a._review_counts_sep.pack(side="right", padx=6, fill="y")
        a._review_counts_label.pack(side="right", padx=(6, 6))
        # Returning from edit: reload GT, recompute, and advance
        if getattr(a, '_review_recompute_on_return', False):
            a._review_recompute_on_return = False
            self._review_reload_gt_and_advance()
        # Returning from annotate (manual tab switch): recompute in
        # case the user added/modified GT annotations ad-hoc.
        elif (a._review_original_image is not None
              and a._review_matches is not None):
            self._review_reload_gt_and_advance()
        # Normal switch: full image load or restore
        elif a.images:
            if a._review_original_image is None:
                # First switch — pick the first reviewable image
                if a._review_filtered_images:
                    a._review_index = a._review_filtered_images[0]
                else:
                    # No images to review
                    self._display_review_image()
                    self._update_review_labels()
                    a._update_status()
                    return
                self._review_load_image()
                a._review_needs_first_zoom = True
            elif not a._review_filtered_images:
                # Returning to review but nothing to review
                a._review_original_image = None
                self._display_review_image()
                self._update_review_labels()
                a._update_status()
                return
            else:
                # Already loaded — preserve det index and zoom
                a._review_cached_scale = None
                self._display_review_image()
            # Zoom to first detection on initial switch
            if a._review_needs_first_zoom:
                a._review_needs_first_zoom = False
                self._review_zoom_to_first_unreviewed()
        self._update_review_labels()
        a._update_status()

    # ══════════════════════════════════════════════════════════════════════════
    #  Coordinate conversion
    # ══════════════════════════════════════════════════════════════════════════

    def _review_image_to_canvas(self, ix, iy):
        a = self.app
        return (ix * a._review_scale + a._review_offset_x,
                iy * a._review_scale + a._review_offset_y)

    def _review_canvas_to_image(self, cx, cy):
        a = self.app
        s = a._review_scale if a._review_scale else 1.0
        return ((cx - a._review_offset_x) / s,
                (cy - a._review_offset_y) / s)

    # ══════════════════════════════════════════════════════════════════════════
    #  Pan / Zoom
    # ══════════════════════════════════════════════════════════════════════════

    def _review_pan_start(self, event):
        a = self.app
        a._review_pan_start_x = event.x
        a._review_pan_start_y = event.y

    def _review_pan_drag(self, event):
        a = self.app
        if a._review_pan_start_x is None:
            return
        dx = event.x - a._review_pan_start_x
        dy = event.y - a._review_pan_start_y
        a._review_offset_x += dx
        a._review_offset_y += dy
        a._review_pan_start_x = event.x
        a._review_pan_start_y = event.y
        self._display_review_image()

    def _review_zoom(self, event):
        a = self.app
        cx, cy = self._review_canvas_to_image(event.x, event.y)
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        new_scale = max(0.05, min(10.0, a._review_scale * factor))
        a._review_offset_x = event.x - cx * new_scale
        a._review_offset_y = event.y - cy * new_scale
        a._review_scale = new_scale
        a._review_cached_scale = None
        self._display_review_image()
        a._update_status()

    def _review_scroll(self, event):
        a = self.app
        if sys.platform == "darwin":
            delta = -event.delta * 2
        else:
            delta = -event.delta // 3
        a._review_offset_y -= delta
        self._display_review_image()

    def _review_hscroll(self, event):
        a = self.app
        if sys.platform == "darwin":
            delta = -event.delta * 2
        else:
            delta = -event.delta // 3
        a._review_offset_x -= delta
        self._display_review_image()

    def _review_scroll_linux(self, event, direction):
        self.app._review_offset_y += direction * 40
        self._display_review_image()

    def _review_hscroll_linux(self, event, direction):
        self.app._review_offset_x += direction * 40
        self._display_review_image()

    def _review_zoom_linux(self, event, direction):
        a = self.app
        cx, cy = self._review_canvas_to_image(event.x, event.y)
        factor = 1.15 if direction > 0 else 1 / 1.15
        new_scale = max(0.05, min(10.0, a._review_scale * factor))
        a._review_offset_x = event.x - cx * new_scale
        a._review_offset_y = event.y - cy * new_scale
        a._review_scale = new_scale
        a._review_cached_scale = None
        self._display_review_image()
        a._update_status()

    # ══════════════════════════════════════════════════════════════════════════
    #  Canvas resize
    # ══════════════════════════════════════════════════════════════════════════

    def _on_review_canvas_configure(self, event=None):
        """Redraw review canvas on window resize (debounced)."""
        a = self.app
        if a._review_original_image is None:
            return
        a._review_cached_scale = None
        if a._review_resize_after_id is not None:
            a.root.after_cancel(a._review_resize_after_id)
        a._review_resize_after_id = a.root.after(
            200, self._finalize_review_resize)

    def _finalize_review_resize(self):
        a = self.app
        a._review_resize_after_id = None
        a._review_cached_scale = None
        self._display_review_image()

    # ══════════════════════════════════════════════════════════════════════════
    #  Toggle visibility
    # ══════════════════════════════════════════════════════════════════════════

    def _on_review_gt_toggled(self):
        """Toggle GT visibility in Review tab."""
        a = self.app
        a._review_show_gt = a._review_gt_var.get()
        self._display_review_image()

    def _on_review_pred_toggled(self):
        """Toggle prediction visibility in Review tab."""
        a = self.app
        a._review_show_pred = a._review_pred_var.get()
        self._display_review_image()

    # ══════════════════════════════════════════════════════════════════════════
    #  Image filter list
    # ══════════════════════════════════════════════════════════════════════════

    def _rebuild_review_image_list(self):
        """Build filtered list of image indices for Review tab.

        Only includes images that have predictions and/or annotations.
        Images with neither are skipped entirely.
        """
        a = self.app
        if not a.images:
            a._review_filtered_images = []
            return
        filtered = []
        has_any_preds = False
        for img_name in a.images:
            stem = os.path.splitext(img_name)[0]
            has_pred = (
                (a.pred_detect_dir and os.path.exists(
                    os.path.join(a.pred_detect_dir, f"{stem}.txt")))
                or (a.pred_segment_dir and os.path.exists(
                    os.path.join(a.pred_segment_dir, f"{stem}.txt")))
            )
            if has_pred:
                has_any_preds = True
        if has_any_preds:
            # Show only images that have prediction files
            for i, img_name in enumerate(a.images):
                stem = os.path.splitext(img_name)[0]
                has_pred = (
                    (a.pred_detect_dir and os.path.exists(
                        os.path.join(a.pred_detect_dir, f"{stem}.txt")))
                    or (a.pred_segment_dir and os.path.exists(
                        os.path.join(a.pred_segment_dir, f"{stem}.txt")))
                )
                if has_pred:
                    filtered.append(i)
        else:
            # No predictions anywhere — show images with annotations
            for i, img_name in enumerate(a.images):
                stem = os.path.splitext(img_name)[0]
                has_annot = (
                    (a.detect_dir and os.path.exists(
                        os.path.join(a.detect_dir, f"{stem}.txt")))
                    or (a.segment_dir and os.path.exists(
                        os.path.join(a.segment_dir, f"{stem}.txt")))
                )
                if has_annot:
                    filtered.append(i)
        a._review_filtered_images = filtered

    def _refresh_review_class_filter(self):
        """Update class filter dropdown values from current class_names."""
        a = self.app
        vals = ["All"] + [f"{cid}: {name}"
                          for cid, name in sorted(a.class_names.items())]
        a._review_class_dd.configure(values=vals)

    # ══════════════════════════════════════════════════════════════════════════
    #  Image loading
    # ══════════════════════════════════════════════════════════════════════════

    def _review_load_image(self):
        """Load image, GT, predictions, and run matching for review."""
        from yololabeler.utils import auto_orient_image

        a = self.app
        if not a.images:
            return
        # Record review time for previous image before loading new one
        a._record_review_time()
        a._review_image_start_time = time.time()
        idx = a._review_index
        if idx < 0 or idx >= len(a.images):
            return
        img_name = a.images[idx]
        img_path = os.path.join(a.image_folder, img_name)

        try:
            pil_img = Image.open(img_path)
            pil_img = auto_orient_image(pil_img)
            pil_img = pil_img.convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image for review: {e}")
            return

        a._review_original_image = pil_img
        a._review_img_w = pil_img.width
        a._review_img_h = pil_img.height

        # Load GT from disk
        a._review_gt_boxes = []
        a._review_gt_polygons = []
        stem = os.path.splitext(img_name)[0]

        detect_path = os.path.join(a.detect_dir, f"{stem}.txt")
        try:
            boxes, _ = parse_detect_labels(
                detect_path, a._review_img_w, a._review_img_h)
            a._review_gt_boxes.extend(boxes)
        except Exception:
            pass

        segment_path = os.path.join(a.segment_dir, f"{stem}.txt")
        try:
            polys, _ = parse_segment_labels(
                segment_path, a._review_img_w, a._review_img_h)
            a._review_gt_polygons.extend(polys)
        except Exception:
            pass

        # Load predictions
        a._review_pred_boxes, a._review_pred_polygons = \
            a._load_predictions(
                img_name, a._review_img_w, a._review_img_h)

        # Run matching
        a._review_matches = a._compute_matches(
            a._review_gt_boxes, a._review_gt_polygons,
            a._review_pred_boxes, a._review_pred_polygons,
            iou_threshold=REVIEW_IOU_THRESHOLD,
            conf_threshold=REVIEW_CONF_THRESHOLD)

        self.engine.rebuild_review_detections()
        self._refresh_review_class_filter()

        # Force geometry update so canvas dimensions are accurate
        a._review_canvas.update_idletasks()

        # Fit image to canvas (use scheduled dimensions if canvas not ready)
        cw = a._review_canvas.winfo_width()
        ch = a._review_canvas.winfo_height()
        if cw < 10 or ch < 10:
            # Canvas not realized yet — schedule a deferred load
            a.root.after(50, self._review_deferred_zoom)
            return
        sx = cw / max(a._review_img_w, 1)
        sy = ch / max(a._review_img_h, 1)
        a._review_scale = min(sx, sy)
        a._review_offset_x = (cw - a._review_img_w * a._review_scale) / 2
        a._review_offset_y = (ch - a._review_img_h * a._review_scale) / 2
        a._review_cached_scale = None

        # Zoom to first unreviewed detection (or first if all reviewed)
        self._review_zoom_to_first_unreviewed()

        self._update_review_labels()

    def _review_deferred_zoom(self):
        """Deferred zoom after canvas geometry is available."""
        a = self.app
        cw = a._review_canvas.winfo_width()
        ch = a._review_canvas.winfo_height()
        if cw < 10 or ch < 10:
            a.root.after(50, self._review_deferred_zoom)
            return
        sx = cw / max(a._review_img_w, 1)
        sy = ch / max(a._review_img_h, 1)
        a._review_scale = min(sx, sy)
        a._review_offset_x = (cw - a._review_img_w * a._review_scale) / 2
        a._review_offset_y = (ch - a._review_img_h * a._review_scale) / 2
        a._review_cached_scale = None
        self._review_zoom_to_first_unreviewed()
        self._update_review_labels()

    def _review_zoom_to_first_unreviewed(self):
        """Zoom to first unreviewed detection, or first if all reviewed."""
        a = self.app
        if a._review_detections:
            img_name = a.images[a._review_index] if a.images else ""
            first_unreviewed = 0
            for i, det in enumerate(a._review_detections):
                if not self.engine.find_reviewed_entry(det, img_name):
                    first_unreviewed = i
                    break
            a._review_detection_idx = first_unreviewed
            self._zoom_to_detection()
        else:
            a._review_detection_idx = 0
            self._display_review_image()

    def _review_reload_gt_and_advance(self):
        """Reload GT from disk after annotate edit, recompute, and advance."""
        a = self.app
        if not a.images:
            return
        img_name = a.images[a._review_index]
        stem = os.path.splitext(img_name)[0]

        # Reload GT boxes
        a._review_gt_boxes = []
        detect_path = os.path.join(a.detect_dir, f"{stem}.txt")
        try:
            boxes, _ = parse_detect_labels(
                detect_path, a._review_img_w, a._review_img_h)
            a._review_gt_boxes.extend(boxes)
        except Exception:
            pass

        # Reload GT polygons
        a._review_gt_polygons = []
        segment_path = os.path.join(a.segment_dir, f"{stem}.txt")
        try:
            polys, _ = parse_segment_labels(
                segment_path, a._review_img_w, a._review_img_h)
            a._review_gt_polygons.extend(polys)
        except Exception:
            pass

        self._review_recompute_and_advance()

    # ══════════════════════════════════════════════════════════════════════════
    #  Detection management
    # ══════════════════════════════════════════════════════════════════════════

    def _current_review_det(self):
        """Return the currently focused detection dict, or None."""
        a = self.app
        if (a._review_detections
                and 0 <= a._review_detection_idx
                < len(a._review_detections)):
            return a._review_detections[a._review_detection_idx]
        return None

    def _refind_detection(self, prev_det):
        """Find the index of prev_det in the current detection list.

        Matches by gt/pred type and index.  Returns 0 if not found.
        """
        a = self.app
        if prev_det is None or not a._review_detections:
            return 0
        gt_t = prev_det.get('gt_type')
        gt_i = prev_det.get('gt_idx')
        pt_t = prev_det.get('pred_type')
        pt_i = prev_det.get('pred_idx')
        for i, d in enumerate(a._review_detections):
            if (d.get('gt_type') == gt_t and d.get('gt_idx') == gt_i
                    and d.get('pred_type') == pt_t
                    and d.get('pred_idx') == pt_i):
                return i
        return 0

    # ══════════════════════════════════════════════════════════════════════════
    #  Zoom to detection / label updates
    # ══════════════════════════════════════════════════════════════════════════

    def _zoom_to_detection(self):
        """Auto-zoom and center the review canvas on the current detection."""
        a = self.app
        # Reset GT/Pred checkboxes to checked on each detection focus
        a._review_gt_var.set(True)
        a._review_show_gt = True
        a._review_pred_var.set(True)
        a._review_show_pred = True
        if not a._review_detections:
            self._display_review_image()
            return
        idx = max(0, min(a._review_detection_idx,
                         len(a._review_detections) - 1))
        det = a._review_detections[idx]
        bbox = det.get('bbox')
        if not bbox:
            self._display_review_image()
            return

        x1, y1, x2, y2 = bbox
        det_w = max(x2 - x1, 1)
        det_h = max(y2 - y1, 1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cw = a._review_canvas.winfo_width() or 800
        ch = a._review_canvas.winfo_height() or 600

        # Detection should fill ~1/3 of canvas (3x context)
        scale_x = cw / (det_w * 3)
        scale_y = ch / (det_h * 3)
        new_scale = max(0.05, min(10.0, min(scale_x, scale_y)))

        a._review_scale = new_scale
        a._review_offset_x = cw / 2 - cx * new_scale
        a._review_offset_y = ch / 2 - cy * new_scale
        a._review_cached_scale = None
        self._display_review_image()

    def _update_review_labels(self):
        """Update all navigation counters and status labels."""
        a = self.app
        # Use main toolbar counter/total for review image nav
        n_filtered = len(a._review_filtered_images)
        if n_filtered and a._review_index in a._review_filtered_images:
            filt_pos = a._review_filtered_images.index(a._review_index) + 1
        else:
            filt_pos = a._review_index + 1 if a.images else 0
        a.counter_entry.delete(0, "end")
        a.counter_entry.insert(0, str(filt_pos))
        a.total_label.configure(text=f"/ {n_filtered}")

        # Image name
        if a.images and 0 <= a._review_index < len(a.images):
            a.image_name_label.configure(
                text=a.images[a._review_index])

        # Detection nav label (in status bar)
        n_dets = len(a._review_detections)
        det_pos = a._review_detection_idx + 1 if n_dets else 0
        det_type = ""
        det_status_text = ""
        if n_dets and a._review_detection_idx < n_dets:
            det = a._review_detections[a._review_detection_idx]
            det_type = det['det_type'].upper() + " "
            # Show per-detection review status
            img_name = a.images[a._review_index] if a.images else ""
            entry = self.engine.find_reviewed_entry(det, img_name) if img_name else None
            if entry:
                action = entry.get("action", "reviewed")
                det_status_text = f"({action})"
            else:
                det_status_text = "(not reviewed)"
        a._review_det_label.configure(text=f"{det_type}{det_pos} / {n_dets}")
        a._review_det_status_label.configure(text=det_status_text)

        # TP/FP/FN counts (in top toolbar)
        matches = a._review_matches
        if matches:
            tp = len(matches.get('tp', []))
            fp = len(matches.get('fp', []))
            fn = len(matches.get('fn', []))
            a._review_counts_label.configure(
                text=f"TP: {tp} | FP: {fp} | FN: {fn}")
        else:
            a._review_counts_label.configure(text="TP: 0 | FP: 0 | FN: 0")

        # Update button text based on detection context
        if n_dets and a._review_detection_idx < n_dets:
            det = a._review_detections[a._review_detection_idx]
            dt = det['det_type']
            if dt == 'fn':
                a._review_accept_btn.configure(text="Keep GT (A)")
                a._review_reject_btn.configure(text="Delete GT (R)")
            elif dt == 'fp':
                a._review_accept_btn.configure(text="Add to GT (A)")
                a._review_reject_btn.configure(text="Dismiss (R)")
            else:  # tp
                a._review_accept_btn.configure(text="Confirm (A)")
                a._review_reject_btn.configure(text="Delete GT (R)")

        # Update zoom text for review tab
        a._update_status()

    # ══════════════════════════════════════════════════════════════════════════
    #  Rendering (absorbed from ReviewRenderer)
    # ══════════════════════════════════════════════════════════════════════════

    def _display_review_image(self):
        """Render image with GT and prediction overlays on review canvas."""
        a = self.app
        c = a._review_canvas
        c.delete("all")

        if a._review_original_image is None:
            c.create_text(
                c.winfo_width() // 2, c.winfo_height() // 2,
                text="No annotations or predictions to review!",
                fill=FG_COLOR, font=(a.font_family, 14))
            return

        s = a._review_scale
        ox, oy = a._review_offset_x, a._review_offset_y
        cw = c.winfo_width() or 800
        ch = c.winfo_height() or 600

        # Visible region in image coords
        ix0 = max(0, int(-ox / s))
        iy0 = max(0, int(-oy / s))
        ix1 = min(a._review_img_w, int((cw - ox) / s) + 1)
        iy1 = min(a._review_img_h, int((ch - oy) / s) + 1)

        crop_w = ix1 - ix0
        crop_h = iy1 - iy0
        if crop_w <= 0 or crop_h <= 0:
            return

        disp_w = max(1, int(crop_w * s))
        disp_h = max(1, int(crop_h * s))

        review_cache_key = (s, ix0, iy0, ix1, iy1)
        if (a._review_cached_scale != review_cache_key
                or a._review_cached_tk_image is None):
            region = a._review_original_image.crop(
                (ix0, iy0, ix1, iy1))
            region = region.resize(
                (disp_w, disp_h), Image.Resampling.BILINEAR)
            a._review_cached_tk_image = ImageTk.PhotoImage(region)
            a._review_cached_scale = review_cache_key

        px = ox + ix0 * s
        py = oy + iy0 * s
        c.create_image(px, py, anchor="nw",
                       image=a._review_cached_tk_image)

        # Determine which detection is focused
        focused_det = None
        if (a._review_detections
                and 0 <= a._review_detection_idx
                < len(a._review_detections)):
            focused_det = a._review_detections[a._review_detection_idx]

        # Build sets of focused GT and pred indices for highlighting
        focused_gt = set()
        focused_pred = set()
        if focused_det:
            if focused_det['gt_type'] is not None:
                focused_gt.add(
                    (focused_det['gt_type'], focused_det['gt_idx']))
            if focused_det['pred_type'] is not None:
                focused_pred.add(
                    (focused_det['pred_type'], focused_det['pred_idx']))

        # Build set of reviewed GT indices from ALL matches (unfiltered)
        reviewed_gt = set()
        img_name = a.images[a._review_index] if a.images else ""
        if img_name and a._review_matches:
            for gt_type, gt_idx, p_type, p_idx, iou, cid, conf in \
                    a._review_matches.get('tp', []):
                det = {'det_type': 'tp', 'class_id': cid,
                       'gt_type': gt_type, 'gt_idx': gt_idx,
                       'pred_type': p_type, 'pred_idx': p_idx}
                if self.engine.find_reviewed_entry(det, img_name):
                    reviewed_gt.add((gt_type, gt_idx))
            for gt_type, gt_idx, cid in a._review_matches.get('fn', []):
                det = {'det_type': 'fn', 'class_id': cid,
                       'gt_type': gt_type, 'gt_idx': gt_idx,
                       'pred_type': None, 'pred_idx': None}
                if self.engine.find_reviewed_entry(det, img_name):
                    reviewed_gt.add((gt_type, gt_idx))

        # Constant line width (1.5pt ~ 2px) regardless of zoom
        line_w = 2
        focused_line_w = 3
        label_size = max(7, min(int(9 * (0.6 + s * 0.4)), 18))
        PRED_COLOR = "#00BFFF"
        FOCUSED_GT_COLOR = "#FFD700"

        REVIEWED_STIPPLE = "gray12"
        PRED_STIPPLE = "gray12"  # fill for FP predictions (no matching GT)

        def _halo_text(x, y, text, fill, **kw):
            halo_text(c, x, y, text, fill, **kw)

        # Determine GT draw format matching prediction format
        has_pred_boxes = bool(a._review_pred_boxes)
        has_pred_polys = bool(a._review_pred_polygons)
        if has_pred_boxes and not has_pred_polys:
            gt_draw_mode = "box"
        elif has_pred_polys and not has_pred_boxes:
            gt_draw_mode = "polygon"
        elif has_pred_boxes and has_pred_polys:
            gt_draw_mode = "both"
        else:
            gt_draw_mode = a.mode

        # ── Draw GT annotations (format matched to predictions) ──
        if a._review_show_gt:
            _deferred_focused_gt = []

            # Draw GT boxes
            if gt_draw_mode in ("box", "both"):
                for i, (x1, y1, x2, y2, cid) in enumerate(
                        a._review_gt_boxes):
                    is_focused = ('box', i) in focused_gt
                    if is_focused:
                        _deferred_focused_gt.append(
                            ('rect', x1, y1, x2, y2, cid))
                        continue
                    if x2 < ix0 or x1 > ix1 or y2 < iy0 or y1 > iy1:
                        continue
                    color = a._get_class_color(cid)
                    is_reviewed = ('box', i) in reviewed_gt
                    cx1, cy1 = self._review_image_to_canvas(x1, y1)
                    cx2, cy2 = self._review_image_to_canvas(x2, y2)
                    if is_reviewed:
                        c.create_rectangle(cx1, cy1, cx2, cy2,
                                           outline=color, width=line_w,
                                           fill=color,
                                           stipple=REVIEWED_STIPPLE)
                    else:
                        c.create_rectangle(cx1, cy1, cx2, cy2,
                                           outline=color, width=line_w,
                                           fill="")
                # Also draw polygon GT as bounding boxes
                if gt_draw_mode == "box":
                    for i, (pts, cid) in enumerate(
                            a._review_gt_polygons):
                        is_focused = ('polygon', i) in focused_gt
                        if is_focused:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            _deferred_focused_gt.append(
                                ('rect', min(xs), min(ys),
                                 max(xs), max(ys), cid))
                            continue
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        if (max(xs) < ix0 or min(xs) > ix1
                                or max(ys) < iy0 or min(ys) > iy1):
                            continue
                        color = a._get_class_color(cid)
                        is_reviewed = ('polygon', i) in reviewed_gt
                        bx1, by1 = min(xs), min(ys)
                        bx2, by2 = max(xs), max(ys)
                        cx1, cy1 = self._review_image_to_canvas(bx1, by1)
                        cx2, cy2 = self._review_image_to_canvas(bx2, by2)
                        if is_reviewed:
                            c.create_rectangle(
                                cx1, cy1, cx2, cy2,
                                outline=color, width=line_w,
                                fill=color, stipple=REVIEWED_STIPPLE)
                        else:
                            c.create_rectangle(
                                cx1, cy1, cx2, cy2,
                                outline=color, width=line_w, fill="")

            # Draw GT polygons
            if gt_draw_mode in ("polygon", "both"):
                for i, (pts, cid) in enumerate(a._review_gt_polygons):
                    is_focused = ('polygon', i) in focused_gt
                    if is_focused:
                        _deferred_focused_gt.append(('poly', pts, cid))
                        continue
                    pxs = [p[0] for p in pts]
                    pys = [p[1] for p in pts]
                    if (max(pxs) < ix0 or min(pxs) > ix1
                            or max(pys) < iy0 or min(pys) > iy1):
                        continue
                    color = a._get_class_color(cid)
                    is_reviewed = ('polygon', i) in reviewed_gt
                    canvas_pts = []
                    for px_pt, py_pt in pts:
                        cx_p, cy_p = self._review_image_to_canvas(
                            px_pt, py_pt)
                        canvas_pts.extend([cx_p, cy_p])
                    if len(canvas_pts) >= 6:
                        if is_reviewed:
                            c.create_polygon(
                                *canvas_pts, outline=color,
                                width=line_w, fill=color,
                                stipple=REVIEWED_STIPPLE)
                        else:
                            c.create_polygon(
                                *canvas_pts, outline=color,
                                width=line_w, fill="")
                # Also draw box GT as polygons
                if gt_draw_mode == "polygon":
                    for i, (x1, y1, x2, y2, cid) in enumerate(
                            a._review_gt_boxes):
                        is_focused = ('box', i) in focused_gt
                        if is_focused:
                            _deferred_focused_gt.append(
                                ('poly', [(x1, y1), (x2, y1),
                                          (x2, y2), (x1, y2)], cid))
                            continue
                        if x2 < ix0 or x1 > ix1 or y2 < iy0 or y1 > iy1:
                            continue
                        color = a._get_class_color(cid)
                        is_reviewed = ('box', i) in reviewed_gt
                        rect_pts = [
                            (x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        canvas_pts = []
                        for px_pt, py_pt in rect_pts:
                            cx_p, cy_p = self._review_image_to_canvas(
                                px_pt, py_pt)
                            canvas_pts.extend([cx_p, cy_p])
                        if is_reviewed:
                            c.create_polygon(
                                *canvas_pts, outline=color,
                                width=line_w, fill=color,
                                stipple=REVIEWED_STIPPLE)
                        else:
                            c.create_polygon(
                                *canvas_pts, outline=color,
                                width=line_w, fill="")

            # ── Draw focused GT last (always on top) ──
            for _dfg in _deferred_focused_gt:
                if _dfg[0] == 'rect':
                    _, fx1, fy1, fx2, fy2, fcid = _dfg
                    cx1, cy1 = self._review_image_to_canvas(fx1, fy1)
                    cx2, cy2 = self._review_image_to_canvas(fx2, fy2)
                    c.create_rectangle(
                        cx1, cy1, cx2, cy2,
                        outline=FOCUSED_GT_COLOR,
                        fill="", width=focused_line_w)
                    name = a.class_names.get(fcid, str(fcid))
                    _halo_text(
                        cx1 + 2, cy1 - 2, anchor="sw",
                        text=f"GT {fcid}: {name}",
                        fill=FOCUSED_GT_COLOR,
                        font=(a.font_family, label_size, "bold"))
                elif _dfg[0] == 'poly':
                    _, fpts, fcid = _dfg
                    canvas_pts = []
                    for px_pt, py_pt in fpts:
                        cx_p, cy_p = self._review_image_to_canvas(
                            px_pt, py_pt)
                        canvas_pts.extend([cx_p, cy_p])
                    if len(canvas_pts) >= 6:
                        c.create_polygon(
                            *canvas_pts,
                            outline=FOCUSED_GT_COLOR,
                            fill="", width=focused_line_w)
                    if fpts:
                        lx, ly = self._review_image_to_canvas(*fpts[0])
                        name = a.class_names.get(fcid, str(fcid))
                        _halo_text(
                            lx + 2, ly - 2, anchor="sw",
                            text=f"GT {fcid}: {name}",
                            fill=FOCUSED_GT_COLOR,
                            font=(a.font_family, label_size, "bold"))

        # ── Draw ONLY the focused prediction (blue) ──
        if a._review_show_pred and focused_det:
            p_type = focused_det.get('pred_type')
            p_idx = focused_det.get('pred_idx')
            if p_type == 'box' and p_idx is not None:
                if 0 <= p_idx < len(a._review_pred_boxes):
                    bx1, by1, bx2, by2, cid, conf = \
                        a._review_pred_boxes[p_idx]
                    cx1, cy1 = self._review_image_to_canvas(bx1, by1)
                    cx2, cy2 = self._review_image_to_canvas(bx2, by2)
                    # FP predictions get a translucent blue fill
                    pred_fill = PRED_COLOR if focused_det.get('det_type') == 'fp' else ""
                    pred_stipple = PRED_STIPPLE if focused_det.get('det_type') == 'fp' else ""
                    c.create_rectangle(
                        cx1, cy1, cx2, cy2,
                        outline=PRED_COLOR, fill=pred_fill,
                        stipple=pred_stipple, width=line_w)
                    name = a.class_names.get(cid, str(cid))
                    _halo_text(
                        cx1 + 2, cy2 + 2, anchor="nw",
                        text=f"Pred {cid}: {name} ({conf:.2f})",
                        fill=PRED_COLOR,
                        font=(a.font_family, label_size, "bold"))
            elif p_type == 'polygon' and p_idx is not None:
                if 0 <= p_idx < len(a._review_pred_polygons):
                    pts, cid, conf = a._review_pred_polygons[p_idx]
                    canvas_pts = []
                    max_cy = 0
                    min_cx = float('inf')
                    for px_pt, py_pt in pts:
                        cx_p, cy_p = self._review_image_to_canvas(
                            px_pt, py_pt)
                        canvas_pts.extend([cx_p, cy_p])
                        if cy_p > max_cy:
                            max_cy = cy_p
                            min_cx = cx_p
                        elif cy_p == max_cy and cx_p < min_cx:
                            min_cx = cx_p
                    if len(canvas_pts) >= 6:
                        pred_fill = PRED_COLOR if focused_det.get('det_type') == 'fp' else ""
                        pred_stipple = PRED_STIPPLE if focused_det.get('det_type') == 'fp' else ""
                        c.create_polygon(
                            *canvas_pts, outline=PRED_COLOR,
                            fill=pred_fill, stipple=pred_stipple,
                            width=line_w)
                    if pts:
                        name = a.class_names.get(cid, str(cid))
                        _halo_text(
                            min_cx + 2, max_cy + 2, anchor="nw",
                            text=f"Pred {cid}: {name} ({conf:.2f})",
                            fill=PRED_COLOR,
                            font=(a.font_family, label_size, "bold"))

        # ── Detection type badge (upper right, semi-transparent bg) ──
        if focused_det and (a._review_show_gt or a._review_show_pred):
            dtype = focused_det['det_type'].upper()
            badge_colors = {
                'TP': '#4CAF50', 'FP': '#EF5350', 'FN': '#FFA726'}
            badge_color = badge_colors.get(dtype, FG_COLOR)
            badge_text = f"[{dtype}]"
            bfnt = tkFont.Font(
                family=a.font_family, size=14, weight="bold")
            tw = bfnt.measure(badge_text)
            th = bfnt.metrics("linespace")
            bx = cw - tw - 20
            by = 10
            c.create_rectangle(
                bx - 6, by - 2, bx + tw + 6, by + th + 4,
                fill="#1A1A1A", outline="#444444", width=1)
            c.create_text(
                bx, by + 2, anchor="nw",
                text=badge_text, fill=badge_color,
                font=(a.font_family, 14, "bold"))

        # ── Review help overlay ──
        if a._review_show_help:
            self._draw_review_help_overlay(c, cw, ch)

    def _draw_review_help_overlay(self, c, cw, ch):
        """Draw a semi-transparent help overlay on the review canvas."""
        help_lines = [
            "\u2500\u2500 Keyboard \u2500\u2500",
            "  a   Accept",
            "        FP \u2192 add prediction to GT",
            "        FN \u2192 keep GT (model missed it)",
            "        TP \u2192 confirm match",
            "  e   Edit (switch to Annotate tab)",
            "        FP \u2192 edit prediction as new GT",
            "        FN \u2192 edit GT annotation",
            "        TP \u2192 edit GT annotation",
            "  h   Toggle this help",
            "  m   Toggle Box / Polygon mode",
            "  r   Reject",
            "        FP \u2192 dismiss prediction",
            "        FN \u2192 delete GT annotation",
            "        TP \u2192 dismiss match",
            "  \u2190 / \u2192   Previous / Next detection",
            "  \u2191 / \u2193   Next / Previous image",
            "",
            "\u2500\u2500 Mouse \u2500\u2500",
            "  Ctrl+Scroll    Zoom at cursor",
            "  Middle-click   Pan (drag)",
            "  Scroll         Pan up / down",
            "  Shift+Scroll   Pan left / right",
        ]
        if sys.platform == "darwin":
            font_family = "Menlo"
        else:
            font_family = "Consolas"
        font_size = 14
        pad = 14
        fnt = tkFont.Font(family=font_family, size=font_size)
        line_height = fnt.metrics("linespace") + 2
        max_text_w = max(fnt.measure(ln) for ln in help_lines)
        block_w = max_text_w + pad * 3
        block_h = len(help_lines) * line_height + pad * 2
        x0, y0 = 10, 10
        c.create_rectangle(x0, y0, x0 + block_w, y0 + block_h,
                           fill="#1A1A1A", outline="#444444", width=1)
        for i, line in enumerate(help_lines):
            c.create_text(x0 + pad, y0 + pad + i * line_height,
                          anchor="nw", text=line, fill=FG_COLOR,
                          font=(font_family, font_size))

    # ══════════════════════════════════════════════════════════════════════════
    #  Navigation
    # ══════════════════════════════════════════════════════════════════════════

    def _review_next_image(self, reset_filters=True):
        a = self.app
        if not a.images:
            return
        if reset_filters:
            # Reset filters to 'all' on manual image change
            a._review_filter_type = "all"
            a._review_type_var.set("All")
            a._review_status_filter = "all"
            a._review_filter_var.set("All")
            a._review_filter_class = "all"
            a._review_class_var.set("All")
        filt = a._review_filtered_images
        if filt:
            try:
                pos = filt.index(a._review_index)
                a._review_index = filt[(pos + 1) % len(filt)]
            except ValueError:
                a._review_index = filt[0]
        else:
            a._review_index = (a._review_index + 1) % len(a.images)
        a._review_original_image = None
        a._review_cached_scale = None
        a._review_cached_tk_image = None
        self._review_load_image()

    def _review_prev_image(self, reset_filters=True):
        a = self.app
        if not a.images:
            return
        if reset_filters:
            # Reset filters to 'all' on manual image change
            a._review_filter_type = "all"
            a._review_type_var.set("All")
            a._review_status_filter = "all"
            a._review_filter_var.set("All")
            a._review_filter_class = "all"
            a._review_class_var.set("All")
        filt = a._review_filtered_images
        if filt:
            try:
                pos = filt.index(a._review_index)
                a._review_index = filt[(pos - 1) % len(filt)]
            except ValueError:
                a._review_index = filt[-1]
        else:
            a._review_index = (a._review_index - 1) % len(a.images)
        a._review_original_image = None
        a._review_cached_scale = None
        a._review_cached_tk_image = None
        self._review_load_image()

    def _review_next_detection(self):
        a = self.app
        if not a._review_detections:
            return
        a._review_detection_idx = (
            (a._review_detection_idx + 1) % len(a._review_detections))
        self._zoom_to_detection()
        self._update_review_labels()

    def _review_prev_detection(self):
        a = self.app
        if not a._review_detections:
            return
        a._review_detection_idx = (
            (a._review_detection_idx - 1) % len(a._review_detections))
        self._zoom_to_detection()
        self._update_review_labels()

    # ══════════════════════════════════════════════════════════════════════════
    #  Filter callbacks
    # ══════════════════════════════════════════════════════════════════════════

    def _on_review_type_changed(self, choice):
        a = self.app
        prev_det = self._current_review_det()
        a._review_filter_type = choice.lower()
        self.engine.rebuild_review_detections()
        a._review_detection_idx = self._refind_detection(prev_det)
        if a._review_detections:
            self._zoom_to_detection()
        else:
            self._display_review_image()
        self._update_review_labels()

    def _on_review_class_changed(self, choice):
        a = self.app
        prev_det = self._current_review_det()
        if choice == "All":
            a._review_filter_class = "all"
        else:
            try:
                a._review_filter_class = int(choice.split(":")[0])
            except (ValueError, IndexError):
                a._review_filter_class = "all"
        self.engine.rebuild_review_detections()
        a._review_detection_idx = self._refind_detection(prev_det)
        if a._review_detections:
            self._zoom_to_detection()
        else:
            self._display_review_image()
        self._update_review_labels()

    def _on_review_filter_changed(self, value):
        """Handle review status filter dropdown change."""
        a = self.app
        prev_det = self._current_review_det()
        mapping = {"All": "all", "Reviewed": "reviewed",
                   "Not Reviewed": "not_reviewed"}
        a._review_status_filter = mapping.get(value, "all")
        self.engine.rebuild_review_detections()
        a._review_detection_idx = self._refind_detection(prev_det)
        if a._review_detections:
            self._zoom_to_detection()
        else:
            self._display_review_image()
        self._update_review_labels()

    # ══════════════════════════════════════════════════════════════════════════
    #  Actions
    # ══════════════════════════════════════════════════════════════════════════

    def _review_accept(self):
        """Accept the current detection.

        TP: Confirm match is correct — step to next.
        FP: Switch to Annotate to draw annotation with
            prediction shown as blue reference overlay.
        FN: Keep GT as-is (model just missed it) — step to next.
        """
        a = self.app
        if not a._review_detections:
            return
        det = a._review_detections[a._review_detection_idx]

        if det['det_type'] == 'fp':
            # FP Accept → switch to Annotate for annotation
            self._switch_to_annotate_for_review(det)
            return

        # TP and FN: no disk changes, just step forward
        self._review_step_next(action="accepted")

    def _review_reject(self):
        """Reject the current detection.

        TP: GT annotation is wrong despite model match — delete GT.
        FP: Prediction is wrong — dismiss (step to next, no disk change).
        FN: GT annotation is wrong — delete GT from disk.
        """
        a = self.app
        if not a._review_detections:
            return
        det = a._review_detections[a._review_detection_idx]

        if det['det_type'] in ('fn', 'tp'):
            # Record action before GT deletion changes the list
            self.engine.record_detection_action(det, "rejected")
            # Backup originals before first destructive edit
            self.engine.backup_original_labels()
            # Delete GT annotation from disk
            gt_type = det['gt_type']
            gt_idx = det['gt_idx']
            if gt_type == 'box' and gt_idx is not None:
                if 0 <= gt_idx < len(a._review_gt_boxes):
                    a._review_gt_boxes.pop(gt_idx)
                    self.engine.save_gt()
            elif gt_type == 'polygon' and gt_idx is not None:
                if 0 <= gt_idx < len(a._review_gt_polygons):
                    a._review_gt_polygons.pop(gt_idx)
                    self.engine.save_gt()
            self._review_recompute_and_advance()
        else:
            # FP reject = dismiss, step to next
            self._review_step_next(action="rejected")

    def _review_edit(self):
        """Switch to Annotate tab for editing, with prediction reference.

        All types: switch to Annotate zoomed to the detection,
        with the prediction (if any) shown as a dashed blue reference.
        """
        a = self.app
        if not a._review_detections:
            return
        det = a._review_detections[a._review_detection_idx]
        self._switch_to_annotate_for_review(det)

    def _switch_to_annotate_for_review(self, det):
        """Switch to Annotate tab from Review, showing prediction reference."""
        a = self.app
        if not a.images:
            return
        # Capture review zoom/offset
        rev_scale = a._review_scale
        rev_ox = a._review_offset_x
        rev_oy = a._review_offset_y

        # Build prediction reference overlay data
        pred_ref = None
        p_type = det.get('pred_type')
        p_idx = det.get('pred_idx')
        if p_type == 'box' and p_idx is not None:
            if 0 <= p_idx < len(a._review_pred_boxes):
                b = a._review_pred_boxes[p_idx]
                pred_ref = {
                    'type': 'box',
                    'coords': (b[0], b[1], b[2], b[3]),
                    'class_id': b[4], 'conf': b[5]}
        elif p_type == 'polygon' and p_idx is not None:
            if 0 <= p_idx < len(a._review_pred_polygons):
                pts, cid, conf = a._review_pred_polygons[p_idx]
                pred_ref = {
                    'type': 'polygon',
                    'coords': list(pts),
                    'class_id': cid, 'conf': conf}

        a._annotate_pred_reference = pred_ref
        a._review_return_pending = True
        a._review_editing_det = det

        # Navigate annotate tab to same image (defer display to avoid flash)
        a._defer_display = True
        a.index = a._review_index
        a._annotate_tab.load_image()
        a._defer_display = False

        # Sync zoom/position from review to annotate
        at = a._annotate_tab
        at.scale = rev_scale
        at.offset_x = rev_ox
        at.offset_y = rev_oy
        at.zoom_index = at._nearest_zoom_index(rev_scale)
        at._cached_scale = None

        # Determine appropriate mode: prefer polygon if polygon labels exist
        gt_type = det.get('gt_type')
        gt_idx = det.get('gt_idx')
        # Always default to polygon when polygon labels are present
        if a.polygons:
            target_mode = 'polygon'
        elif gt_type == 'polygon':
            target_mode = 'polygon'
        elif gt_type == 'box':
            target_mode = 'box'
        elif det.get('pred_type') == 'polygon':
            target_mode = 'polygon'
        elif det.get('pred_type') == 'box':
            target_mode = 'box'
        else:
            target_mode = None

        if target_mode == 'polygon':
            if a.mode != 'polygon':
                a.mode = 'polygon'
                a.mode_btn.configure(text="Mode: Polygon \u2b21")
                a.stream_btn.configure(state="normal")
                a.snap_btn.configure(state="normal")
            if gt_idx is not None and 0 <= gt_idx < len(a.polygons):
                a._selected_polygon_idx = gt_idx
        elif target_mode == 'box':
            if a.mode != 'box':
                a.mode = 'box'
                a.mode_btn.configure(text="Mode: Box \u25ad")
                a.current_polygon = []
                a._selected_polygon_idx = None
                a.stream_btn.configure(state="disabled")
                a.snap_btn.configure(state="disabled")

        # Switch tab and display with correct zoom
        a.tabview.set("Annotate")
        a._on_tab_changed()

    def _review_confirm_dialog(self):
        """Show confirmation dialog after annotation while in review flow."""
        a = self.app
        dialog = ctk.CTkToplevel(a.root)
        dialog.title("Confirm GT Edits")
        dialog.geometry("360x150")
        dialog.resizable(False, False)
        dialog.transient(a.root)
        dialog.grab_set()
        dialog.configure(fg_color=BG_COLOR)

        # Center on parent
        dialog.update_idletasks()
        px = a.root.winfo_x() + (a.root.winfo_width() - 360) // 2
        py = a.root.winfo_y() + (a.root.winfo_height() - 150) // 2
        dialog.geometry(f"+{px}+{py}")

        ctk.CTkLabel(
            dialog,
            text="Save this annotation to the GT dataset?",
            font=(a.font_family, 13, "bold"),
            text_color=FG_COLOR,
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            dialog,
            text="Accept saves the edits and returns to review.\n"
                 "Redo undoes the last edits so you can redo.",
            font=(a.font_family, 11),
            text_color=FG_COLOR,
        ).pack(pady=(0, 12))

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=(0, 10))

        def on_accept():
            dialog.destroy()
            self._save_and_return_to_review()

        def on_redo():
            dialog.destroy()
            # Undo the last annotation so user can redraw
            a._annotate_tab.undo_last()

        ctk.CTkButton(
            btn_frame, text="\u2713 Accept", width=120,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(a.font_family, 12, "bold"),
            command=on_accept,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            btn_frame, text="\u21ba Redo", width=120,
            fg_color=SI_PERSIMMON, hover_color="#F0A77B",
            text_color=FG_COLOR, font=(a.font_family, 12, "bold"),
            command=on_redo,
        ).pack(side="left")

    def _save_and_return_to_review(self):
        """Save current annotations and return to Review tab."""
        a = self.app
        a._annotate_tab.save_annotations()
        # Record the action for the detection that was being edited
        if hasattr(a, '_review_editing_det') and a._review_editing_det:
            self.engine.record_detection_action(
                a._review_editing_det, "edited")
            a._review_editing_det = None
        a._annotate_pred_reference = None
        a._review_return_pending = False
        # Switch back to Review and recompute+advance (GT was modified)
        a._review_recompute_on_return = True
        a.tabview.set("Review")
        a._on_tab_changed()

    # ══════════════════════════════════════════════════════════════════════════
    #  Advance logic
    # ══════════════════════════════════════════════════════════════════════════

    def _review_advance_or_switch_type(self):
        """Advance to next image, or switch match-type filter if unreviewed
        detections remain.

        If the current filter is not 'all', checks other match types for
        unreviewed detections on the same image before advancing.
        """
        a = self.app
        if a._review_filter_type != "all" and a._review_matches:
            img_name = a.images[a._review_index] if a.images else ""
            type_order = ["tp", "fp", "fn"]
            for mtype in type_order:
                if mtype == a._review_filter_type:
                    continue
                entries = a._review_matches.get(mtype, [])
                if not entries:
                    continue
                # Build a quick det dict for each entry to check review status
                for entry in entries:
                    if mtype == "tp":
                        gt_type, gt_idx, p_type, p_idx, iou, cid, conf = entry
                        det = {'det_type': 'tp', 'class_id': cid, 'conf': conf,
                               'iou': iou, 'gt_type': gt_type, 'gt_idx': gt_idx,
                               'pred_type': p_type, 'pred_idx': p_idx}
                    elif mtype == "fp":
                        p_type, p_idx, cid, conf = entry
                        det = {'det_type': 'fp', 'class_id': cid, 'conf': conf,
                               'iou': None, 'gt_type': None, 'gt_idx': None,
                               'pred_type': p_type, 'pred_idx': p_idx}
                    else:  # fn
                        gt_type, gt_idx, cid = entry
                        det = {'det_type': 'fn', 'class_id': cid, 'conf': None,
                               'iou': None, 'gt_type': gt_type, 'gt_idx': gt_idx,
                               'pred_type': None, 'pred_idx': None}
                    if not self.engine.find_reviewed_entry(det, img_name):
                        # Found unreviewed det in another type — switch filter
                        a._review_filter_type = mtype
                        a._review_type_var.set(mtype.upper())
                        self.engine.rebuild_review_detections()
                        a._review_detection_idx = 0
                        # Jump to first unreviewed in new list
                        for i, d in enumerate(a._review_detections):
                            if not self.engine.find_reviewed_entry(d, img_name):
                                a._review_detection_idx = i
                                break
                        self._zoom_to_detection()
                        self._update_review_labels()
                        return
        # Also check with "all" filter to catch any missed detections
        if a._review_filter_type != "all" and a._review_matches:
            img_name = a.images[a._review_index] if a.images else ""
            saved_type = a._review_filter_type
            a._review_filter_type = "all"
            self.engine.rebuild_review_detections()
            has_unreviewed = False
            for d in a._review_detections:
                if not self.engine.find_reviewed_entry(d, img_name):
                    has_unreviewed = True
                    break
            if has_unreviewed:
                a._review_type_var.set("All")
                a._review_detection_idx = 0
                for i, d in enumerate(a._review_detections):
                    if not self.engine.find_reviewed_entry(d, img_name):
                        a._review_detection_idx = i
                        break
                self._zoom_to_detection()
                self._update_review_labels()
                return
            # Restore and advance
            a._review_filter_type = saved_type
            self.engine.rebuild_review_detections()
        self._review_next_image(reset_filters=False)

    def _review_step_next(self, action="accepted"):
        """Record action and advance to the next unreviewed detection.

        Used after non-modifying actions (TP accept, FN accept, FP reject).
        The action parameter is stored in review_state for tracking.
        """
        a = self.app
        if not a._review_detections:
            self.engine.check_image_review_complete()
            self._review_advance_or_switch_type()
            return
        # Record the action for the current detection
        det = a._review_detections[a._review_detection_idx]
        self.engine.record_detection_action(det, action)
        # Advance to next unreviewed detection (skip already-reviewed ones)
        img_name = a.images[a._review_index] if a.images else ""
        start = a._review_detection_idx
        n = len(a._review_detections)
        for offset in range(1, n + 1):
            next_idx = start + offset
            if next_idx >= n:
                # Reached end of list — check if all reviewed
                self.engine.check_image_review_complete()
                self._review_advance_or_switch_type()
                return
            next_det = a._review_detections[next_idx]
            if not self.engine.find_reviewed_entry(next_det, img_name):
                # Found next unreviewed detection
                a._review_detection_idx = next_idx
                self._zoom_to_detection()
                self._update_review_labels()
                return

    def _review_recompute_and_advance(self):
        """Recompute matches after GT modification and advance.

        Used after modifying actions (FP accept/add, TP/FN reject/delete).
        """
        a = self.app
        a._review_matches = a._compute_matches(
            a._review_gt_boxes, a._review_gt_polygons,
            a._review_pred_boxes, a._review_pred_polygons,
            iou_threshold=REVIEW_IOU_THRESHOLD,
            conf_threshold=REVIEW_CONF_THRESHOLD)
        self.engine.rebuild_review_detections()

        if not a._review_detections:
            self.engine.check_image_review_complete()
            self._review_advance_or_switch_type()
            return

        # Try to find next unreviewed detection from current position
        img_name = a.images[a._review_index] if a.images else ""
        found = False
        for i in range(len(a._review_detections)):
            if not self.engine.find_reviewed_entry(
                    a._review_detections[i], img_name):
                a._review_detection_idx = i
                found = True
                break
        if not found:
            # All reviewed in current filter after recompute
            self.engine.check_image_review_complete()
            self._review_advance_or_switch_type()
            return
        self._zoom_to_detection()
        self._update_review_labels()
