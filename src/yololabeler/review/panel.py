"""ReviewPanel: the status-bar strip, queue stepping, filters and threshold (spec 5.1)."""

from __future__ import annotations

import os
import tkinter as tk

import customtkinter as ctk

from yololabeler.predictions.store import load_predictions
from yololabeler.review.engine import build_queue, flag_markers, match_document, shape_statuses

# Palette constants duplicated from gui.py to avoid a circular import
FG_COLOR = "#E0E0E0"
BG_COLOR = "#1E1E1E"
ACCENT = "#507754"
ACCENT_HOVER = "#608864"
ENTRY_BG = "#2A2A2A"
BORDER_COLOR = "#3A3A3A"
SI_GREEN = "#507754"
REVIEW_IOU_THRESHOLD = 0.5  # matches compute_matches's own default (matching.py:79)


class ReviewPanel:
    """Owns the review strip and the queue over the current image."""

    def __init__(self, app):
        self.app = app
        self.engine = app._review

    # ── widgets ────────────────────────────────────────────────────────────

    def _combo(self, parent, var, values, width, command):
        """Build a read-only dropdown in the shared dark styling."""
        a = self.app
        return ctk.CTkComboBox(
            parent, variable=var, values=values, width=width, command=command,
            font=(a.font_family, 11), dropdown_font=(a.font_family, 11),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, text_color=FG_COLOR,
            dropdown_fg_color=BG_COLOR, dropdown_text_color=FG_COLOR,
            dropdown_hover_color=ACCENT, state="readonly")

    def _label(self, parent, text, **kw):
        """Build a strip label in the shared dark styling."""
        return ctk.CTkLabel(parent, text=text, font=(self.app.font_family, 11),
                            text_color=FG_COLOR, **kw)

    def _button(self, parent, text, width, command, bold=True):
        """Build a strip button in the shared dark styling."""
        a = self.app
        return ctk.CTkButton(
            parent, text=text, width=width, command=command, fg_color=SI_GREEN,
            hover_color=ACCENT_HOVER, text_color=FG_COLOR,
            font=(a.font_family, 11, "bold" if bold else "normal"))

    def build(self, left, centre, right):
        """Create the strip in the status bar's left, centre and right columns.

        Left holds the prediction toggle, threshold, filters and the item stepper
        with its "FP 2 / 16  not reviewed" readout between the arrows; centre
        holds Accept, Edit and Reject; the TP/FP/FN counts are packed on the right,
        to the left of whatever the caller has already packed there.
        """
        a = self.app
        a._pred_var = tk.BooleanVar(value=a._review_show_pred)
        a._pred_cb = ctk.CTkCheckBox(
            left, text="Predictions", variable=a._pred_var, width=1,
            font=(a.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER_COLOR,
            command=a._on_pred_toggled)
        a._pred_cb.pack(side="left", padx=(0, 10))

        self._label(left, "Conf").pack(side="left", padx=(0, 2))
        self.conf_entry = ctk.CTkEntry(left, width=44, font=(a.font_family, 11),
                                       fg_color=ENTRY_BG, border_color=BORDER_COLOR,
                                       text_color=FG_COLOR, justify="center")
        self.conf_entry.pack(side="left", padx=(0, 8))
        self.conf_entry.bind("<Return>", self._on_conf_enter)
        self.conf_entry.bind("<FocusOut>", lambda e: self._show_threshold())
        self._label(left, "Review status").pack(side="left", padx=(0, 2))
        self.status_var = tk.StringVar(value="All")
        self.status_dd = self._combo(left, self.status_var,
                                     ["All", "Not reviewed", "Reviewed", "Flagged"], 100,
                                     self.on_status_changed)
        self.status_dd.pack(side="left", padx=(0, 8))
        self._label(left, "Type").pack(side="left", padx=(0, 2))
        self.type_var = tk.StringVar(value="All")
        self.type_dd = self._combo(left, self.type_var, ["All", "FP", "FN", "TP"], 64,
                                   self.on_type_changed)
        self.type_dd.pack(side="left", padx=(0, 8))
        self.prev_item_btn = self._button(left, "◀", 28, lambda: self.step(-1), bold=False)
        self.prev_item_btn.pack(side="left")
        # Fixed width so the next arrow does not jump as the readout changes length.
        self.item_label = self._label(left, "", width=130, anchor="center")
        self.item_label.pack(side="left", padx=2)
        self.next_item_btn = self._button(left, "▶", 28, lambda: self.step(1), bold=False)
        self.next_item_btn.pack(side="left")

        self.accept_btn = self._button(centre, "Accept (A)", 96, a.accept_item)
        self.accept_btn.pack(side="left", padx=(0, 6))
        self.edit_btn = self._button(centre, "Edit (E)", 84, a.edit_pair)
        self.edit_btn.pack(side="left", padx=(0, 6))
        self.reject_btn = self._button(centre, "Reject (R)", 96, a.reject_item)
        self.reject_btn.pack(side="left")

        a._status_sep_right(right)
        self.counts_label = self._label(right, "TP 0  FP 0  FN 0")
        self.counts_label.pack(side="right", padx=(0, 6))
        self._show_threshold()

    # ── loading and refresh ────────────────────────────────────────────────

    def load_predictions_for_current_image(self):
        """Read the prediction files for the current image unless it is blind."""
        a = self.app
        img_name = a.images[a.index]
        a.predictions, a.predictions_rejected = [], []
        store = a._stats_store
        a.predictions_blind = store.is_blind(img_name) and store.completion(img_name) is None
        if a.predictions_blind:
            return
        stem = os.path.splitext(img_name)[0]
        a.predictions, a.predictions_rejected = load_predictions(
            a.pred_detect_dir, a.pred_segment_dir, stem, a.img_width, a.img_height)
        a._register_class_ids({p.class_id for p in a.predictions})
        dropped = self.engine.migrate_centre_entries(
            img_name, a.predictions, a.img_width, a.img_height)
        if dropped:
            a.show_banner(
                f"{dropped} old review entries matched no current prediction and were dropped.")

    def refresh(self, keep_focus=True):
        """Rerun matching and rebuild the queue; called after every document change."""
        a = self.app
        if a.document is None or a.predictions_blind or not a.predictions:
            a.queue, a.matches, a.shape_statuses = [], {}, None
            a.flag_markers = (flag_markers(a.document, [], {},
                                           self.engine.open_flag_keys(a.images[a.index]))
                              if a.images and a.document is not None else {})
            # Blind mode empties the queue without touching a single verdict, so
            # recomputing the status from it would report every image not_started.
            if a.images and not a.predictions_blind:
                self.engine.update_img_status(a.images[a.index])
            self.update_labels()
            a._annotate_tab.display_image()
            return
        focused = self.current_item() if keep_focus else None
        previous = focused.key if focused else None
        a.matches = match_document(a.document, a.predictions, REVIEW_IOU_THRESHOLD,
                                   a.conf_threshold)
        open_keys = self.engine.open_flag_keys(a.images[a.index])
        a.queue = build_queue(a.document, a.predictions, a.matches, a.verdicts,
                              a._review_filter_type, a._review_filter_class,
                              a._review_status_filter, open_keys)
        a.shape_statuses = shape_statuses(a.document, a.predictions, a.matches, a.verdicts)
        a.flag_markers = flag_markers(a.document, a.predictions, a.matches, open_keys)
        a.queue_index = 0
        if previous is not None:
            for i, item in enumerate(a.queue):
                if item.key == previous:
                    a.queue_index = i
                    break
        self.update_labels()
        if a.images:
            self.engine.update_img_status(a.images[a.index])
        a._annotate_tab.display_image()

    def first_unreviewed(self):
        """Index of the first queue item with no verdict, 0 when every item has one."""
        a = self.app
        for i, item in enumerate(a.queue):
            if item.key not in a.verdicts:
                return i
        return 0

    def next_unreviewed_after(self, path, key):
        """Queue index of the first unreviewed item after key along path, wrapping.

        path is the list of item keys in path order taken before the last
        verdict; a key no longer in the queue is skipped. Falls back to
        first_unreviewed when nothing along the path qualifies.
        """
        a = self.app
        start = path.index(key) + 1 if key in path else 0
        positions = {item.key: i for i, item in enumerate(a.queue)}
        for candidate in path[start:] + path[:start]:
            if candidate in positions and candidate not in a.verdicts:
                return positions[candidate]
        return self.first_unreviewed()

    # ── stepping ───────────────────────────────────────────────────────────

    def focus_item(self, index, zoom=True, switch_class=True):
        """Focus a queue item, set mode (and optionally class) to match it, zoom to it (spec 4.4)."""
        a = self.app
        if not a.queue:
            return
        a.queue_index = index % len(a.queue)
        item = a.queue[a.queue_index]
        if switch_class:
            a._select_class_by_id(item.class_id)
        kind = item.prediction.kind if item.prediction else item.annotation.kind
        if a.mode != kind:
            a._set_mode(kind)
        if zoom:
            shape = item.prediction or item.annotation
            xs = [p[0] for p in shape.points]
            ys = [p[1] for p in shape.points]
            a._annotate_tab.zoom_to_bbox(min(xs), min(ys), max(xs), max(ys))
        self.update_labels()
        a._annotate_tab.display_image()

    def step(self, delta):
        """Move the focus delta items along the queue, wrapping at both ends."""
        if self.app.queue:
            self.focus_item(self.app.queue_index + delta)

    def current_item(self):
        """The focused QueueItem, or None when the queue is empty."""
        a = self.app
        if a.queue and 0 <= a.queue_index < len(a.queue):
            return a.queue[a.queue_index]
        return None

    # ── threshold and filters ──────────────────────────────────────────────

    def _show_threshold(self):
        """Write the current confidence threshold into the entry.

        Forces the entry back to "normal" first: a disabled Tk entry silently
        ignores delete/insert, and _init_folder calls this while conf_entry may
        still be disabled from a blind image in the previously open folder.
        """
        self.conf_entry.configure(state="normal")
        self.conf_entry.delete(0, "end")
        self.conf_entry.insert(0, f"{self.app.conf_threshold:.2f}")

    def _on_conf_enter(self, event=None):
        """Apply the typed confidence threshold, reverting anything out of range."""
        try:
            value = float(self.conf_entry.get())
        except ValueError:
            self._show_threshold()
            return
        if not 0.0 <= value <= 1.0:
            self._show_threshold()
            return
        self.set_threshold(value)
        self.app.canvas.focus_set()

    def set_threshold(self, value):
        """Store the confidence threshold on the state and the engine, then rematch."""
        self.app.conf_threshold = value
        self.engine.conf_threshold = value
        self._show_threshold()
        self.refresh()

    def on_type_changed(self, choice):
        """Filter the queue by match type."""
        self.app._review_filter_type = choice.lower()
        self.refresh(keep_focus=False)
        self.app.canvas.focus_set()

    def on_status_changed(self, choice):
        """Filter the queue by verdict presence."""
        mapping = {"All": "all", "Reviewed": "reviewed", "Not reviewed": "not_reviewed",
                   "Flagged": "flagged"}
        self.app._review_status_filter = mapping.get(choice, "all")
        self.refresh(keep_focus=False)
        self.app.canvas.focus_set()

    # ── labels ─────────────────────────────────────────────────────────────

    def update_labels(self):
        """Refresh the item counter, the counts and the accept/reject button state."""
        a = self.app
        item = self.current_item()
        if a.predictions_blind:
            self.item_label.configure(text="Blind")
            self.counts_label.configure(text="")
            state = "disabled"
        elif item is None:
            self.item_label.configure(text="No items")
            state = "disabled"
        else:
            verdict = a.verdicts.get(item.key)
            status = verdict["action"] if verdict else "not reviewed"
            self.item_label.configure(
                text=f"{item.kind.upper()} {a.queue_index + 1} / {len(a.queue)}  {status}")
            state = "normal"
        self.accept_btn.configure(state=state)
        self.edit_btn.configure(state=state)
        self.reject_btn.configure(state=state)
        filter_state = "disabled" if a.predictions_blind else "readonly"
        control_state = "disabled" if a.predictions_blind else "normal"
        self.type_dd.configure(state=filter_state)
        self.status_dd.configure(state=filter_state)
        self.conf_entry.configure(state=control_state)
        self.prev_item_btn.configure(state=control_state)
        self.next_item_btn.configure(state=control_state)
        if not a.predictions_blind:
            m = a.matches or {}
            pending = sum(1 for qi in a.queue if qi.key not in a.verdicts)
            notes = [f"{pending} not reviewed"] if pending else []
            if a.flag_markers:
                notes.append(f"{len(a.flag_markers)} flagged")
            suffix = f"  ({', '.join(notes)})" if notes else ""
            self.counts_label.configure(
                text=f"TP {len(m.get('tp', []))}  FP {len(m.get('fp', []))}  "
                     f"FN {len(m.get('fn', []))}{suffix}")
        a.complete_cb.configure(text="Complete")
        a._update_status()
