"""ReviewPanel: the status-bar strip, queue stepping, filters and threshold (spec 5.1)."""

from __future__ import annotations

import os
import tkinter as tk

import customtkinter as ctk

from yololabeler.predictions.store import load_predictions
from yololabeler.review.engine import build_queue, match_document

# Palette constants duplicated from gui.py to avoid a circular import
FG_COLOR = "#E0E0E0"
BG_COLOR = "#1E1E1E"
ACCENT = "#507754"
ACCENT_HOVER = "#608864"
ENTRY_BG = "#2A2A2A"
BORDER_COLOR = "#3A3A3A"
SI_GREEN = "#507754"
REVIEW_IOU_THRESHOLD = 0.60  # spec 4.3; constant, displayed beside the conf entry


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

    def build(self, si):
        """Create the strip inside the status bar's inner frame."""
        a = self.app
        self.frame = ctk.CTkFrame(si, fg_color="transparent")
        self.frame.pack(side="left", fill="x", expand=True)
        left = ctk.CTkFrame(self.frame, fg_color="transparent")
        left.pack(side="left", expand=True, fill="x")
        centre = ctk.CTkFrame(self.frame, fg_color="transparent")
        centre.pack(side="left", expand=True, fill="x")
        right = ctk.CTkFrame(self.frame, fg_color="transparent")
        right.pack(side="left", expand=True, fill="x")

        self._label(left, "Class").pack(side="left", padx=(0, 2))
        self.class_var = tk.StringVar(value="All")
        self.class_dd = self._combo(left, self.class_var, ["All"], 90, self.on_class_changed)
        self.class_dd.pack(side="left", padx=(0, 4))
        self._label(left, "Type").pack(side="left", padx=(0, 2))
        self.type_var = tk.StringVar(value="All")
        self._combo(left, self.type_var, ["All", "FP", "FN", "TP"], 70,
                    self.on_type_changed).pack(side="left", padx=(0, 4))
        self._label(left, "Status").pack(side="left", padx=(0, 2))
        self.status_var = tk.StringVar(value="All")
        self._combo(left, self.status_var, ["All", "Not reviewed", "Reviewed"], 110,
                    self.on_status_changed).pack(side="left", padx=(0, 4))
        self.gt_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="GT", variable=self.gt_var, width=40,
                        font=(a.font_family, 11), text_color=FG_COLOR, fg_color=ACCENT,
                        hover_color=ACCENT_HOVER, border_color=BORDER_COLOR,
                        command=self.on_gt_toggled).pack(side="left", padx=(4, 2))
        self.pred_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Pred", variable=self.pred_var, width=45,
                        font=(a.font_family, 11), text_color=FG_COLOR, fg_color=ACCENT,
                        hover_color=ACCENT_HOVER, border_color=BORDER_COLOR,
                        command=self.on_pred_toggled).pack(side="left", padx=(0, 4))
        self._label(left, "Conf").pack(side="left", padx=(4, 2))
        self.conf_entry = ctk.CTkEntry(left, width=50, font=(a.font_family, 11),
                                       fg_color=ENTRY_BG, border_color=BORDER_COLOR,
                                       text_color=FG_COLOR, justify="center")
        self.conf_entry.pack(side="left", padx=(0, 2))
        self.conf_entry.bind("<Return>", self._on_conf_enter)
        self.conf_entry.bind("<FocusOut>", lambda e: self._show_threshold())
        self._label(left, f"IoU {REVIEW_IOU_THRESHOLD:.2f}").pack(side="left", padx=(2, 4))

        self.accept_btn = self._button(centre, "Accept (A)", 110, a.accept_item)
        self.accept_btn.pack(side="left", padx=(0, 4))
        self.reject_btn = self._button(centre, "Reject (R)", 110, a.reject_item)
        self.reject_btn.pack(side="left", padx=(0, 4))

        self.item_label = self._label(right, "", width=150, anchor="w")
        self.item_label.pack(side="left", padx=(0, 4))
        self._button(right, "◀", 30, lambda: self.step(-1), bold=False).pack(side="left")
        self._button(right, "▶", 30, lambda: self.step(1),
                     bold=False).pack(side="left", padx=(2, 6))
        self.counts_label = self._label(right, "TP 0  FP 0  FN 0")
        self.counts_label.pack(side="left")
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
        if a.document is None or a.predictions_blind:
            a.queue, a.matches = [], {}
            self.update_labels()
            return
        focused = self.current_item() if keep_focus else None
        previous = focused.key if focused else None
        a.matches = match_document(a.document, a.predictions, REVIEW_IOU_THRESHOLD,
                                   a.conf_threshold)
        a.queue = build_queue(a.document, a.predictions, a.matches, a.verdicts,
                              a._review_filter_type, a._review_filter_class,
                              a._review_status_filter)
        a.queue_index = 0
        if previous is not None:
            for i, item in enumerate(a.queue):
                if item.key == previous:
                    a.queue_index = i
                    break
        self.update_labels()
        a._annotate_tab.display_image()

    def first_unreviewed(self):
        """Index of the first queue item with no verdict, 0 when every item has one."""
        a = self.app
        for i, item in enumerate(a.queue):
            if item.key not in a.verdicts:
                return i
        return 0

    # ── stepping ───────────────────────────────────────────────────────────

    def focus_item(self, index, zoom=True):
        """Focus a queue item, set class and mode to match it, and zoom to it (spec 4.4)."""
        a = self.app
        if not a.queue:
            return
        a.queue_index = index % len(a.queue)
        item = a.queue[a.queue_index]
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
        """Write the current confidence threshold into the entry."""
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

    def on_class_changed(self, choice):
        """Filter the queue by class id."""
        a = self.app
        a._review_filter_class = "all" if choice == "All" else int(choice.split(":")[0])
        self.refresh(keep_focus=False)

    def on_status_changed(self, choice):
        """Filter the queue by verdict presence."""
        mapping = {"All": "all", "Reviewed": "reviewed", "Not reviewed": "not_reviewed"}
        self.app._review_status_filter = mapping.get(choice, "all")
        self.refresh(keep_focus=False)

    def on_gt_toggled(self):
        """Show or hide the focused annotation layer."""
        self.app._review_show_gt = self.gt_var.get()
        self.app._annotate_tab.display_image()

    def on_pred_toggled(self):
        """Show or hide the prediction layer."""
        self.app._review_show_pred = self.pred_var.get()
        self.app._annotate_tab.display_image()

    def refresh_class_filter(self):
        """Rebuild the class dropdown from the current class registry."""
        a = self.app
        self.class_dd.configure(values=["All"] + [f"{cid}: {name}"
                                                  for cid, name in sorted(a.class_names.items())])

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
        self.reject_btn.configure(state=state)
        if not a.predictions_blind:
            m = a.matches or {}
            self.counts_label.configure(
                text=f"TP {len(m.get('tp', []))}  FP {len(m.get('fp', []))}  "
                     f"FN {len(m.get('fn', []))}")
        pending = sum(1 for item in a.queue if item.key not in a.verdicts)
        a.complete_cb.configure(text=f"Complete ({pending} not reviewed)" if pending else "Complete")
        a._update_status()
