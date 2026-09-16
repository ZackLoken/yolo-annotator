"""YoloLabeler main application window.

Composes one AppState, one AnnotationEngine, one ReviewEngine, one
AnnotateTab and one ReviewPanel over a single canvas, and owns the
toolbar, status bar, class registry, folder loading and session stats.
"""

import os
import sys
import json
import time
import getpass
import datetime
from collections import namedtuple
import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image, ImageTk

import customtkinter as ctk
import shutil

from yololabeler.state import AppState
from yololabeler.state_io import AnnotationStats, read_json_or_quarantine
from yololabeler.annotation.engine import AnnotationEngine
from yololabeler.annotation.tab import AnnotateTab
from yololabeler.keybindings import KEY_BINDINGS
from yololabeler.predictions.importers import import_predictions, FORMATS
from yololabeler.predictions.store import read_manifest
from yololabeler.review.engine import ReviewEngine, apply_accept, apply_reject
from yololabeler.review.panel import ReviewPanel
from yololabeler.utils import (
    suppress_tk_mac_warnings, _load_custom_fonts, _get_font_family,
    ASSETS_DIR, is_image_file,
)

# Lightweight event object for synthesised clicks
_SynthEvent = namedtuple('_SynthEvent', ['x', 'y'])


# ── Dark Theme Palette ─────────────────────────────────────────────────────────
BG_COLOR = "#1E1E1E"       # dark gray background
FG_COLOR = "#E0E0E0"       # light gray text
ACCENT = "#507754"          # SI green — buttons, highlights
ACCENT_HOVER = "#608864"    # slightly lighter green for hovers
CANVAS_BG = "#2D2D2D"      # canvas background
ENTRY_BG = "#2A2A2A"       # entry/combo background
BORDER_COLOR = "#3A3A3A"   # subtle borders

# SI Brand colors, offered as swatches in the class colour picker
SI_GREEN = "#507754"
SI_WATER_BLUE = "#83A0BA"
SI_WOOD = "#C7B299"
SI_STEM_GREEN = "#7E8F60"
SI_LAKE_BLUE = "#367A8A"
SI_MULBERRY = "#996967"
SI_PERSIMMON = "#E6976B"
SI_ELDERBERRY = "#2A194E"
SI_SAGE = "#889E6E"
SI_LEAF_GREEN = "#6F9382"

# High-contrast default class colors — visible against natural/outdoor scenes
DEFAULT_CLASS_COLORS = [
    "#FF0000",  # Red
    "#00FFFF",  # Cyan
    "#FFFF00",  # Yellow
    "#FF00FF",  # Magenta
    "#FF8C00",  # Orange
    "#00FF00",  # Lime
    "#FFFFFF",  # White
    "#4169E1",  # Royal Blue
    "#FF69B4",  # Hot Pink
    "#00CED1",  # Dark Turquoise
]


class YoloLabeler:

    # Attributes transparently forwarded to self._state (AppState).
    # Everything else lives directly on self (GUI widgets, view transform, etc.).
    _STATE_ATTRS = frozenset({
        # Paths
        'image_folder', 'labels_dir', 'detect_dir', 'segment_dir', 'state_dir',
        'pred_detect_dir', 'pred_segment_dir',
        # Image list & current image
        'images', 'index', 'original_image', 'img_width', 'img_height',
        # Annotations
        'document', 'load_errors', 'verdicts',
        'current_polygon', 'mode',
        'start_x', 'start_y', 'rect',
        # Class registry
        'class_names', 'class_colors', 'active_class',
        # Undo / redo
        '_undo_stack', '_redo_stack', '_vertex_redo_stack',
        # Snap
        'snap_enabled',
        # Completion / filter
        '_completed_images', '_active_filter', '_filtered_indices',
        # Spatial index
        '_poly_bboxes', '_poly_bboxes_dirty',
        # Interaction
        '_dragging_vertex', '_drag_orig_pos',
        '_selected_annotation_id', '_hovered_annotation_id',
        '_stream_mode', '_stream_active', '_last_stream_pos',
        # Review data
        'queue', 'queue_index',
        'predictions', 'predictions_rejected', 'predictions_blind',
        'matches', 'conf_threshold',
        '_review_filter_type', '_review_filter_class', '_review_status_filter',
        '_review_show_gt', '_review_show_pred', '_review_state',
        '_annotation_visible',
        # Stats & session
        '_stats',
        '_current_user', '_session_start',
        '_image_start_time',
        '_session_annotated_images', '_session_images',
        '_session_loaded_counts', '_session_add_counts', '_session_total_adds',
        # Misc state
        'show_help', 'banner_text',
    })

    def __getattr__(self, name):
        """Forward data-attribute reads to AppState."""
        if name in type(self)._STATE_ATTRS:
            return getattr(object.__getattribute__(self, '_state'), name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Forward data-attribute writes to AppState."""
        if name in type(self)._STATE_ATTRS:
            state = self.__dict__.get('_state')
            if state is not None:
                setattr(state, name, value)
                return
        object.__setattr__(self, name, value)

    def __init__(self, root, image_folder=None, class_names=None):
        self.root = root
        # All annotation/review data defaults live in AppState; only values
        # that differ from those defaults are set here.
        object.__setattr__(self, '_state', AppState())
        object.__setattr__(self, '_engine', AnnotationEngine(self._state))
        object.__setattr__(self, '_review',
                           ReviewEngine(self._state, on_error=self.show_banner))
        self.image_folder = image_folder or ""
        self._constructor_class_names = dict(class_names) if class_names else {}
        self.class_names = dict(self._constructor_class_names)
        self._current_user = getpass.getuser()
        self._session_start = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # GUI-only handles (not part of AppState)
        self._timer_after_id = None
        self._logo_image = None  # keep a reference so Tk does not drop the image
        self._app_icon_image = None  # same, for the window/taskbar icon
        self._stats_store = AnnotationStats()
        self._stats = self._stats_store.data

        # ── Build GUI ──
        _load_custom_fonts()
        self.font_family = _get_font_family()
        self._set_window_icon()
        self._build_toolbar()
        object.__setattr__(self, '_review_panel', ReviewPanel(self))
        self._build_status_bar()

        # ── Canvas ──
        self._canvas_frame = ctk.CTkFrame(self.root, fg_color=BG_COLOR)
        self._canvas_frame.pack(fill="both", expand=True)
        object.__setattr__(self, '_annotate_tab', AnnotateTab(self))
        self._annotate_tab.build(self._canvas_frame)
        self.canvas = self._annotate_tab.canvas

        self._setup_bindings()

        # ── Start ──
        if self.image_folder:
            self._init_folder(self.image_folder)
            self.root.after(100, self._annotate_tab.load_image)
        else:
            self.root.after(100, self._show_welcome)

        self._start_timer_display()

    # ──────────────────────────────────────────────────────────────────────────
    #  Toolbar (top)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        self.toolbar = ctk.CTkFrame(self.root, fg_color=BG_COLOR, height=44, 
                                    corner_radius=0)
        self.toolbar.pack(side="top", fill="x")
        self.toolbar.pack_propagate(False)

        inner = ctk.CTkFrame(self.toolbar, fg_color="transparent")
        inner.pack(fill="x", padx=6, pady=4)

        # ── LEFT: Logo | Open Folder (always visible) ──
        self._load_logo(inner)

        self.open_btn = ctk.CTkButton(
            inner, text="\U0001f4c2 Open Folder", width=120,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12),
            command=self._open_folder)
        self.open_btn.pack(side="left", padx=(8, 8))

        self.import_btn = ctk.CTkButton(
            inner, text="Import predictions", width=140,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12),
            command=self._open_import_form)
        self.import_btn.pack(side="left", padx=(0, 8))

        # ── RIGHT: always-visible nav (pack first so it stays rightmost) ──
        self._toolbar_right = ctk.CTkFrame(inner, fg_color="transparent")
        self._toolbar_right.pack(side="right")

        self.next_btn = ctk.CTkButton(
            self._toolbar_right, text="Next \u25b6", width=70,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12, "bold"),
            command=lambda: self._annotate_tab.next_image())
        self.next_btn.pack(side="right", padx=(2, 4))

        self.total_label = ctk.CTkLabel(
            self._toolbar_right, text="/ 0", font=(self.font_family, 12),
            text_color=FG_COLOR)
        self.total_label.pack(side="right", padx=(2, 4))

        self.counter_entry = ctk.CTkEntry(
            self._toolbar_right, width=55, font=(self.font_family, 12),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            text_color=FG_COLOR, justify="center")
        self.counter_entry.pack(side="right", padx=(2, 0))
        self.counter_entry.bind("<Return>", self._on_counter_enter)
        self.counter_entry.bind("<FocusOut>", self._on_counter_focus_out)

        self.prev_btn = ctk.CTkButton(
            self._toolbar_right, text="\u25c0 Prev", width=70,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12, "bold"),
            command=lambda: self._annotate_tab.prev_image())
        self.prev_btn.pack(side="right", padx=(4, 2))

        self.image_name_label = ctk.CTkLabel(
            self._toolbar_right, text="", font=(self.font_family, 11),
            text_color="#AAAAAA")
        self.image_name_label.pack(side="right", padx=(4, 2))

        # ── CENTER: Three annotation groups ──
        self._toolbar_center = ctk.CTkFrame(inner, fg_color="transparent")
        self._toolbar_center.pack(side="left", fill="x", expand=True)

        _tb_g1 = ctk.CTkFrame(self._toolbar_center, fg_color="transparent")
        _tb_g1.pack(side="left", expand=True, fill="x")

        _tb_g2 = ctk.CTkFrame(self._toolbar_center, fg_color="transparent")
        _tb_g2.pack(side="left", expand=True, fill="x")

        _tb_g3 = ctk.CTkFrame(self._toolbar_center, fg_color="transparent")
        _tb_g3.pack(side="left", expand=True, fill="x")

        # ── Group 1: Color Picker | Class DD | Visible ──
        self.color_btn = tk.Button(
            _tb_g1, text="  ", width=2, relief="flat",
            borderwidth=1, command=self._pick_class_color,
            bg=self._get_class_color(self.active_class),
            activebackground=self._get_class_color(self.active_class))
        self.color_btn.pack(side="left", padx=(2, 4))

        self.class_var = tk.StringVar()
        self.class_dropdown = ctk.CTkComboBox(
            _tb_g1, variable=self.class_var, width=180,
            font=(self.font_family, 11),
            dropdown_font=(self.font_family, 11),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, dropdown_fg_color=BG_COLOR,
            dropdown_text_color=FG_COLOR,
            dropdown_hover_color=ACCENT,
            state="readonly",
            command=self._on_class_selected)
        self.class_dropdown.pack(side="left", padx=(0, 4))
        self._refresh_class_dropdown()

        self._visible_var = tk.BooleanVar(value=True)
        self._visible_cb = ctk.CTkCheckBox(
            _tb_g1, text="Visible",
            variable=self._visible_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR,
            command=self._on_visible_toggled)
        self._visible_cb.pack(side="left", padx=(0, 4))

        # ── Group 2: Mode | Stream | Snap ──
        self.mode_btn = ctk.CTkButton(
            _tb_g2, text="Mode: Polygon \u2b21", width=120,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_mode)
        self.mode_btn.pack(side="left", padx=(0, 4))

        self.stream_btn = ctk.CTkButton(
            _tb_g2, text="Stream: Off", width=95,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_stream)
        self.stream_btn.pack(side="left", padx=(0, 4))

        self.snap_btn = ctk.CTkButton(
            _tb_g2, text="Snap: Off", width=80,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_snap)
        self.snap_btn.pack(side="left", padx=(0, 4))

        # ── Group 3: Complete | Status DD ──
        self._complete_var = tk.BooleanVar(value=False)
        self.complete_cb = ctk.CTkCheckBox(
            _tb_g3, text="Complete",
            variable=self._complete_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR,
            command=self._on_complete_toggled)
        self.complete_cb.pack(side="left", padx=(0, 4))

        self._blind_var = tk.BooleanVar(value=False)
        self.blind_cb = ctk.CTkCheckBox(
            _tb_g3, text="Blind", variable=self._blind_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=BORDER_COLOR,
            command=self._on_blind_toggled)
        self.blind_cb.pack(side="left", padx=(0, 4))

        ctk.CTkLabel(_tb_g3, text="Status:",
                     font=(self.font_family, 11),
                     text_color=FG_COLOR).pack(side="left", padx=(4, 2))

        self.filter_var = tk.StringVar(value="All")
        self.filter_dropdown = ctk.CTkComboBox(
            _tb_g3, variable=self.filter_var, width=130,
            values=["All", "Complete", "Partial", "Unannotated"],
            font=(self.font_family, 11),
            dropdown_font=(self.font_family, 11),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, dropdown_fg_color=BG_COLOR,
            dropdown_text_color=FG_COLOR,
            dropdown_hover_color=ACCENT,
            state="readonly",
            command=self._on_filter_changed)
        self.filter_dropdown.pack(side="left", padx=(0, 4))

    def _toolbar_sep(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=28,
                           fg_color=BORDER_COLOR)
        sep.pack(side="left", padx=6, fill="y")

    # ──────────────────────────────────────────────────────────────────────────
    #  Status bar (bottom)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        self.status_bar = ctk.CTkFrame(self.root, fg_color=BG_COLOR,
                                       height=32, corner_radius=0)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.pack_propagate(False)

        si = ctk.CTkFrame(self.status_bar, fg_color="transparent")
        si.pack(fill="x", padx=8, pady=2)

        # ── Right side: Zoom | Time | User ──
        # Packed before the strip so a narrow window squeezes the strip, not these.
        self.status_user = ctk.CTkLabel(
            si, text=f"User: {self._current_user}",
            font=(self.font_family, 11), text_color=FG_COLOR)
        self.status_user.pack(side="right", padx=(6, 6))

        self._status_sep_right(si)

        self.status_time = ctk.CTkLabel(
            si, text="Image time: 0:00", font=(self.font_family, 11),
            text_color=FG_COLOR)
        self.status_time.pack(side="right", padx=(6, 6))

        self._status_sep_right(si)

        self.status_zoom = ctk.CTkLabel(
            si, text="Zoom: 100%", font=(self.font_family, 11),
            text_color=FG_COLOR)
        self.status_zoom.pack(side="right", padx=(6, 6))

        self._review_panel.build(si)

    def _status_sep_right(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=20,
                           fg_color=BORDER_COLOR)
        sep.pack(side="right", padx=6, fill="y")

    def _update_status(self):
        pct = int(self._annotate_tab.scale * 100)
        self.status_zoom.configure(text=f"Zoom: {pct}%")

    def _on_visible_toggled(self):
        """Toggle annotation visibility on the canvas."""
        self._annotation_visible = self._visible_var.get()
        if self.original_image is not None:
            self._annotate_tab.display_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Icon and logo
    # ──────────────────────────────────────────────────────────────────────────
    def _set_window_icon(self):
        """Set the window/taskbar icon from the bundled app_icon assets.

        CustomTkinter's CTk schedules its own titlebar-icon override 200ms
        after construction unless iconbitmap was called first, so iconphoto
        alone is silently overwritten on Windows; iconbitmap must run too.
        """
        self._app_icon_image = ImageTk.PhotoImage(
            Image.open(os.path.join(ASSETS_DIR, "app_icon.png")))
        self.root.iconphoto(True, self._app_icon_image)
        if sys.platform.startswith("win"):
            self.root.iconbitmap(os.path.join(ASSETS_DIR, "app_icon.ico"))

    def _load_logo(self, parent):
        logo_path = os.path.join(ASSETS_DIR, "si_logo.png")
        if os.path.exists(logo_path):
            try:
                logo = Image.open(logo_path)
                h = 40
                ratio = h / logo.height
                w = int(logo.width * ratio)
                logo = logo.resize((w, h), Image.Resampling.LANCZOS)
                self._logo_image = ImageTk.PhotoImage(logo)
                lbl = tk.Label(parent, image=self._logo_image, bg=BG_COLOR)
                lbl.pack(side="left", padx=(4, 4))
            except Exception:
                self._logo_fallback(parent)
        else:
            self._logo_fallback(parent)

    def _logo_fallback(self, parent):
        ctk.CTkLabel(parent, text="YoloLabeler",
                     font=(self.font_family, 13, "bold"),
                     text_color=ACCENT).pack(side="left", padx=(4, 4))

    # ──────────────────────────────────────────────────────────────────────────
    #  Bindings
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def ACTIONS(self):
        """Every Binding.action in KEY_BINDINGS mapped to the method that runs it."""
        tab, panel = self._annotate_tab, self._review_panel
        actions = {
            "prev_image": tab.prev_image, "next_image": tab.next_image,
            "prev_item": lambda: panel.step(-1), "next_item": lambda: panel.step(1),
            "accept": self.accept_item, "reject": self.reject_item,
            "edit_pair": self.edit_pair, "fit": tab.fit_to_window,
            "zoom_item": lambda: panel.focus_item(self.queue_index),
            "toggle_mode": self._toggle_mode, "toggle_snap": self._toggle_snap_key,
            "toggle_stream": self._toggle_stream_key,
            "undo": self.undo, "redo": self.redo, "save": self.save_now,
            "click": self._click_at_cursor, "escape": self._on_escape,
            "help": tab.toggle_help, "rename_class": self._rename_class_dialog,
        }
        for n in range(10):
            actions[f"class_{n}"] = lambda n=n: self._select_class_by_id(n)
        return actions

    def _bind_keys(self):
        """Bind every action in KEY_BINDINGS to its method; the table is the only source."""
        actions = self.ACTIONS
        for binding in KEY_BINDINGS:
            method = actions[binding.action]
            for seq in binding.sequences:
                if seq.startswith("<Command-") and sys.platform != "darwin":
                    continue
                self.root.bind(seq, lambda e, m=method: self._key_action(m))

    def _setup_bindings(self):
        """Bind the key table and the window protocol; canvas bindings live in AnnotateTab."""
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        c = self.canvas
        c.focus_set()
        c.bind("<Enter>", lambda e: c.focus_set())

    def _text_widget_focused(self):
        """True when keyboard focus is in an entry or combobox, so letter keys must type."""
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return True
        return focused is not None and isinstance(focused.master, ctk.CTkComboBox)

    def _key_action(self, action):
        """Run a bound action unless the user is typing into a widget, clearing any banner first."""
        if self._text_widget_focused():
            return
        self.clear_banner()
        action()

    def _toggle_snap_key(self):
        """Toggle snapping, which only applies in polygon mode."""
        if self.mode == "polygon":
            self._toggle_snap()

    def _toggle_stream_key(self):
        """Toggle vertex streaming, which only applies in polygon mode."""
        if self.mode == "polygon":
            self._toggle_stream()

    def _click_at_cursor(self):
        """Left click at the current pointer position, so the spacebar places a vertex."""
        cx = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        cy = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        self._annotate_tab.on_button_press(_SynthEvent(cx, cy))

    def undo(self):
        """Undo the last annotation change, persist the restored verdicts, rebuild the queue."""
        self._annotate_tab.undo_last()
        self._review_panel.refresh()

    def redo(self):
        """Redo the last undone change, persist the restored verdicts, rebuild the queue."""
        self._annotate_tab.redo_last()
        self._review_panel.refresh()

    def _act_on_item(self, apply):
        """Shared body of accept and reject: undo point, mutate, save, record, refresh."""
        item = self._review_panel.current_item()
        if item is None or self.predictions_blind:
            return
        self._engine.push_undo()
        action, _ = apply(item)
        # Labels first, so a crash cannot leave a verdict for an unwritten change.
        self.save_current()
        self._review.record_verdict(self.images[self.index], item, action, self._current_user)
        self._mark_image_annotated()
        self._review_panel.refresh(keep_focus=False)
        self._review_panel.focus_item(self._review_panel.first_unreviewed())

    def accept_item(self):
        """Accept the focused queue item, promoting an fp prediction into an annotation."""
        self._act_on_item(lambda item: apply_accept(self.document, item, self._current_user))

    def reject_item(self):
        """Reject the focused queue item, removing its annotation if it has one."""
        self._act_on_item(lambda item: apply_reject(self.document, item))

    def edit_pair(self):
        """Select the focused item's annotation so its vertices can be edited (spec 5.3)."""
        item = self._review_panel.current_item()
        if item is None or item.annotation is None:
            return
        self._annotate_tab.select_annotation(item.annotation.id)

    def save_current(self):
        """Save the current image's document and stats. Returns None or an error message."""
        if not self.images or self.document is None:
            return None
        if self.load_errors:
            # A read-only image still records its status and timing.
            self._save_stats()
            return None
        self._review.backup_original_labels()
        error = self._annotate_tab.save_annotations()
        if error:
            message = f"{error}. Fix it and press Ctrl+S."
            self.show_banner(message)
            return message
        self._save_stats()
        return None

    def save_now(self):
        """Save the current image on demand (Ctrl+S)."""
        self.save_current()

    def go_to_image(self, index, reset_filters=True):
        """Save, then load another image (spec 5.2). Returns False when the save failed.

        reset_filters resets the review Type/Status filters to "all" for a
        manual navigation step; the class filter is left untouched since it is
        sticky across pages.
        """
        if not self.images:
            return False
        if self.save_current():
            return False
        self._record_image_time()
        self.banner_text = None
        if reset_filters:
            self._review_filter_type = "all"
            self._review_status_filter = "all"
        self.index = index % len(self.images)
        self._annotate_tab.load_image()
        return True

    def show_banner(self, text):
        """One canvas message, replacing any previous one (spec 7.1)."""
        self.banner_text = text
        self._annotate_tab.display_image()

    def clear_banner(self):
        """Dismiss the current banner, if any."""
        if self.banner_text is not None:
            self.banner_text = None
            self._annotate_tab.display_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Mode toggle
    # ──────────────────────────────────────────────────────────────────────────
    def _set_mode(self, mode):
        """Switch annotation mode and update the toolbar buttons to match.

        Leaving polygon mode discards any in-progress polygon, selection,
        drag and streaming state, since none of them apply to boxes.
        """
        self.mode = mode
        if mode == "polygon":
            self.mode_btn.configure(text="Mode: Polygon \u2b21")
            self.stream_btn.configure(state="normal")
            self.snap_btn.configure(state="normal")
        else:
            self.mode_btn.configure(text="Mode: Box \u25ad")
            self.current_polygon = []
            self._dragging_vertex = None
            self._drag_orig_pos = None
            self._selected_annotation_id = None
            self._stream_mode = False
            self._stream_active = False
            self.stream_btn.configure(text="Stream: Off", state="disabled")
            self.snap_btn.configure(state="disabled")

    def _toggle_mode(self, event=None):
        self._set_mode("polygon" if self.mode == "box" else "box")
        self._annotate_tab.display_image()
        self.update_title()
        self._update_status()

    # ──────────────────────────────────────────────────────────────────────────
    #  Stream toggle (v key or button)
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_stream(self, event=None):
        self._stream_mode = not self._stream_mode
        if self._stream_mode:
            self.stream_btn.configure(text="Stream: On")
        else:
            self.stream_btn.configure(text="Stream: Off")
            self._stream_active = False
            self._last_stream_pos = None

    # ──────────────────────────────────────────────────────────────────────────
    #  Snap toggle
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_snap(self, event=None):
        self.snap_enabled = not self.snap_enabled
        if self.snap_enabled:
            self.snap_btn.configure(text="Snap: On")
        else:
            self.snap_btn.configure(text="Snap: Off")

    # ──────────────────────────────────────────────────────────────────────────
    #  Folder initialisation
    # ──────────────────────────────────────────────────────────────────────────
    def _init_folder(self, folder):
        self.image_folder = folder
        self.images = sorted([f for f in os.listdir(folder) if is_image_file(f)])
        print(f"[YoloLabeler] Opened folder: {folder} ({len(self.images)} images)")
        self.labels_dir = os.path.join(folder, "labels")
        self.detect_dir = os.path.join(self.labels_dir, "detect")
        self.segment_dir = os.path.join(self.labels_dir, "segment")
        os.makedirs(self.detect_dir, exist_ok=True)
        os.makedirs(self.segment_dir, exist_ok=True)

        # State directory (annotation_stats, review_stats, classes)
        self.state_dir = os.path.join(folder, "state")
        os.makedirs(self.state_dir, exist_ok=True)
        self._migrate_state_files()

        # Prediction directories
        self.pred_detect_dir = os.path.join(folder, "predictions", "detect")
        self.pred_segment_dir = os.path.join(folder, "predictions", "segment")
        os.makedirs(self.pred_detect_dir, exist_ok=True)
        os.makedirs(self.pred_segment_dir, exist_ok=True)

        # Reset the registry, load this folder's classes, then merge the constructor's
        self.class_names = {}
        self.class_colors = {}
        moved = self._load_classes_json()
        if moved:
            self.show_banner(
                f"classes.json could not be read and was moved to "
                f"{os.path.basename(moved)}. Starting a new one.")
        for cid, name in self._constructor_class_names.items():
            self.class_names.setdefault(cid, name)
        self._refresh_class_dropdown()
        self._update_color_btn()

        self._load_stats()
        self._load_completed_from_stats()
        self._load_review_state()
        self.conf_threshold = self._review.conf_threshold
        self._review_panel._show_threshold()
        # Prepopulate image_status for every image in the folder
        store = self._stats_store
        for img_name in self.images:
            if img_name not in store.data["image_status"]:
                status = "partial" if self._has_annotations(img_name) else "unannotated"
                store.set_image_status(img_name, status)
        self._save_stats()
        self._rebuild_filter()
        self.index = 0

        # Auto-detect default mode: polygon if segment labels exist,
        # else box if detect-only labels exist
        has_seg = any(
            os.path.exists(os.path.join(
                self.segment_dir, f"{os.path.splitext(img)[0]}.txt"))
            for img in self.images[:50])
        if not has_seg:
            has_det = any(
                os.path.exists(os.path.join(
                    self.detect_dir, f"{os.path.splitext(img)[0]}.txt"))
                for img in self.images[:50])
            if has_det:
                self._set_mode("box")

        # Persist classes (JSON plus the constructor's list, ensures the file exists)
        self._save_classes_file()

    def _migrate_state_files(self):
        """Move legacy JSON files from image folder root into state/."""
        for name in ("annotation_stats.json", "review_stats.json",
                     "review_state.json", "classes.json"):
            old = os.path.join(self.image_folder, name)
            if not os.path.exists(old):
                continue
            # For review_state.json (legacy name), migrate to review_stats.json
            dest_name = "review_stats.json" if name == "review_state.json" else name
            new = os.path.join(self.state_dir, dest_name)
            if os.path.exists(new):
                # state/ already has the file — skip (don't overwrite)
                continue
            shutil.move(old, new)
            print(f"[YoloLabeler] Migrated {name} → state/{dest_name}")

    def _has_annotations(self, img_name):
        stem = os.path.splitext(img_name)[0]
        for label_dir in (self.detect_dir, self.segment_dir):
            if label_dir is None:
                continue
            label_path = os.path.join(label_dir, f"{stem}.txt")
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            return True
            except Exception:
                pass
        return False

    def _show_welcome(self):
        """Show the open-folder prompt on the canvas until a folder is loaded."""
        self.show_canvas_message('Click "Open Folder" to load images')

    def show_canvas_message(self, text):
        """Centre one message on an empty canvas, for failures with no image to draw on."""
        self.root.title("YoloLabeler")
        self.original_image = None
        canvas = self.canvas
        canvas.delete("all")
        self._annotate_tab._cached_scale = None
        self._annotate_tab._cached_tk_image = None
        cw = canvas.winfo_width() or 1200
        ch = canvas.winfo_height() or 800
        canvas.create_text(
            cw // 2, ch // 2, text=text,
            fill=FG_COLOR, font=(self.font_family, 16),
            tags="welcome")
        # add="+" keeps the canvas's own <Configure> resize handler bound.
        canvas.bind("<Configure>", self._reposition_welcome, add="+")

    @staticmethod
    def _reposition_welcome(event):
        """Keep welcome text centered on resize; a no-op once the text is gone."""
        canvas = event.widget
        items = canvas.find_withtag("welcome")
        if items:
            canvas.coords(items[0], canvas.winfo_width() // 2,
                          canvas.winfo_height() // 2)

    # ──────────────────────────────────────────────────────────────────────────
    #  Open folder
    # ──────────────────────────────────────────────────────────────────────────
    def _open_folder(self):
        with suppress_tk_mac_warnings():
            new_folder = filedialog.askdirectory(
                title="Select Folder of Images")
        if not new_folder:
            return

        if self.images:
            self._record_image_time()
            if self.save_current():
                return
            self._end_session()
            self._save_stats()

        self._init_folder(new_folder)

        if not self.images:
            self.show_canvas_message(f"No images found in {new_folder}")
            return

        self._session_start = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self._session_annotated_images = set()
        self._session_images = {}
        self._session_loaded_counts = {}
        self._session_add_counts = {}
        self._session_total_adds = 0
        self._annotate_tab.load_image()

    def _run_import(self, source_dir, fmt, class_id, model_name):
        """Convert and load predictions; returns an error for the form or None."""
        if not self.image_folder:
            return "Open an image folder first."
        if not model_name.strip():
            return "A model name is required."
        try:
            result = import_predictions(source_dir, self.image_folder, fmt,
                                        model_name.strip(), class_id, self._current_user)
        except (ValueError, OSError) as e:
            return str(e)
        self.show_banner(result.summary())
        if self.images and self.document is not None:
            self._review_panel.load_predictions_for_current_image()
            self._review_panel.refresh(keep_focus=False)
        return None

    def _open_import_form(self):
        """A form, not a warning: source folder, format, class id, model name (spec 4.2)."""
        form = ctk.CTkToplevel(self.root)
        form.title("Import predictions")
        form.configure(fg_color=BG_COLOR)
        form.transient(self.root)
        form.grab_set()
        pad = {"padx": 12, "pady": 4}

        def entry(label, width=260):
            row = ctk.CTkFrame(form, fg_color="transparent")
            row.pack(fill="x", **pad)
            ctk.CTkLabel(row, text=label, width=110, anchor="w", font=(self.font_family, 11),
                         text_color=FG_COLOR).pack(side="left")
            e = ctk.CTkEntry(row, width=width, font=(self.font_family, 11), fg_color=ENTRY_BG,
                             border_color=BORDER_COLOR, text_color=FG_COLOR)
            e.pack(side="left")
            return row, e

        src_row, src_entry = entry("Source folder", 200)
        ctk.CTkButton(src_row, text="Browse", width=60, fg_color=ENTRY_BG, hover_color=ACCENT_HOVER,
                      text_color=FG_COLOR, font=(self.font_family, 11),
                      command=lambda: (src_entry.delete(0, "end"),
                                       src_entry.insert(0, filedialog.askdirectory(
                                           title="Prediction files") or ""))).pack(side="left", padx=(6, 0))
        fmt_row = ctk.CTkFrame(form, fg_color="transparent")
        fmt_row.pack(fill="x", **pad)
        ctk.CTkLabel(fmt_row, text="Format", width=110, anchor="w", font=(self.font_family, 11),
                     text_color=FG_COLOR).pack(side="left")
        fmt_var = tk.StringVar(value=FORMATS[2])
        ctk.CTkComboBox(fmt_row, variable=fmt_var, values=list(FORMATS), width=200,
                        font=(self.font_family, 11), fg_color=ENTRY_BG, border_color=BORDER_COLOR,
                        button_color=ACCENT, text_color=FG_COLOR, dropdown_fg_color=BG_COLOR,
                        dropdown_text_color=FG_COLOR, state="readonly").pack(side="left")
        _, class_entry = entry("Class id (JSON only)", 60)
        class_entry.insert(0, "0")
        _, model_entry = entry("Model name")
        message = ctk.CTkLabel(form, text="", font=(self.font_family, 11), text_color=FG_COLOR,
                               wraplength=380, justify="left")
        message.pack(fill="x", **pad)

        def submit():
            fmt = fmt_var.get()
            class_id = None
            if fmt == "bur_detect_json":
                try:
                    class_id = int(class_entry.get())
                except ValueError:
                    message.configure(text="Class id must be a whole number.")
                    return
            error = self._run_import(src_entry.get().strip(), fmt, class_id, model_entry.get())
            if error:
                message.configure(text=error)
            else:
                form.destroy()

        ctk.CTkButton(form, text="Import predictions", width=160, fg_color=ACCENT,
                      hover_color=ACCENT_HOVER, text_color=FG_COLOR,
                      font=(self.font_family, 12), command=submit).pack(pady=(4, 12))

    # ──────────────────────────────────────────────────────────────────────────
    #  Quit
    # ──────────────────────────────────────────────────────────────────────────
    def _quit(self):
        if self.image_folder and self.images:
            error = self.save_current()
            if error and not self._confirm_quit_without_saving():
                return
            self._record_image_time()
            self._end_session()
            self._save_stats()
        if self._timer_after_id:
            self.root.after_cancel(self._timer_after_id)
        if self._annotate_tab._resize_after_id:
            self.root.after_cancel(self._annotate_tab._resize_after_id)
        self.root.destroy()

    def _confirm_quit_without_saving(self):
        """The one modal: a save failed on quit (spec 7.2). True means quit anyway."""
        count = len(self.document.annotations) if self.document else 0
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Save failed")
        dialog.configure(fg_color=BG_COLOR)
        dialog.transient(self.root)
        dialog.grab_set()
        result = {"quit": False}
        ctk.CTkLabel(dialog, text=self.banner_text or "Could not save.",
                     font=(self.font_family, 12), text_color=FG_COLOR,
                     wraplength=380).pack(padx=16, pady=(16, 6))
        ctk.CTkLabel(dialog, text=f"Quitting now loses {count} annotations on this image.",
                     font=(self.font_family, 11), text_color=FG_COLOR).pack(padx=16, pady=(0, 12))
        row = ctk.CTkFrame(dialog, fg_color="transparent")
        row.pack(pady=(0, 14))

        def retry():
            if self.save_current() is None:
                result["quit"] = True
                dialog.destroy()

        def quit_anyway():
            result["quit"] = True
            dialog.destroy()

        ctk.CTkButton(row, text="Retry", width=110, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      text_color=FG_COLOR, command=retry).pack(side="left", padx=(0, 10))
        ctk.CTkButton(row, text="Quit without saving", width=160, fg_color=ENTRY_BG,
                      hover_color=ACCENT_HOVER, text_color=FG_COLOR,
                      command=quit_anyway).pack(side="left")
        dialog.wait_window()
        return result["quit"]

    def _on_escape(self, event=None):
        if self.mode == "polygon":
            if self._stream_active:
                self._stream_active = False
                self._last_stream_pos = None
                self._annotate_tab.display_image()
                return
            if self.current_polygon:
                self.current_polygon = []
                self._annotate_tab.display_image()
                return
            if self._selected_annotation_id is not None:
                self._selected_annotation_id = None
                self._annotate_tab._clear_drag_state()
                self._annotate_tab.display_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Time tracking
    # ──────────────────────────────────────────────────────────────────────────
    def _stats_path(self):
        if self.image_folder:
            return os.path.join(self.state_dir, "annotation_stats.json")
        return None

    def _load_stats(self):
        """Load annotation_stats.json into _stats_store, quarantining it if corrupt."""
        path = self._stats_path()
        if path is None:
            self._stats_store, moved = AnnotationStats(), None
        else:
            self._stats_store, moved = AnnotationStats.load(path)
        self._stats = self._stats_store.data
        if moved:
            self.show_banner(
                f"annotation_stats.json could not be read and was moved to "
                f"{os.path.basename(moved)}. Starting a new one.")

    def _save_stats(self):
        path = self._stats_path()
        if not path:
            return
        try:
            self._stats_store.save(path)
        except OSError as e:
            self.show_banner(f"Could not save annotation_stats.json: {e.strerror or e}")

    # ── Review state persistence ─────────────────────────────────────────────

    def _load_review_state(self):
        """Load review_stats.json, quarantining it if corrupt."""
        moved = self._review.load_review_state()
        if moved:
            self.show_banner(
                f"review_stats.json could not be read and was moved to "
                f"{os.path.basename(moved)}. Starting a new one.")

    def _mark_image_annotated(self):
        """Call whenever user creates/modifies an annotation."""
        if self.images:
            img_name = self.images[self.index]
            self._session_annotated_images.add(img_name)
            if self._image_start_time is None:
                self._image_start_time = time.time()

    def _record_annotation_added(self):
        """Increment gross-add counter for the current image (box or polygon creation only)."""
        if self.images:
            img_name = self.images[self.index]
            self._session_add_counts[img_name] = (
                self._session_add_counts.get(img_name, 0) + 1)
            self._session_total_adds += 1

    def _record_image_time(self):
        if self._image_start_time is None or not self.images:
            return
        elapsed = time.time() - self._image_start_time
        self._image_start_time = None

        img_name = self.images[self.index]
        if img_name not in self._session_annotated_images:
            return

        entry = self._session_images.setdefault(
            img_name, {"session_seconds": 0.0,
                       "loaded_annotation_count": self._session_loaded_counts.get(img_name, 0),
                       "annotations_added": 0,
                       "final_annotation_count": 0})
        entry["session_seconds"] += elapsed

        count = len(self.document.annotations) if self.document else 0
        entry["final_annotation_count"] = count
        adds = self._session_add_counts.get(img_name, 0)
        entry["annotations_added"] += adds
        self._session_add_counts[img_name] = 0

        if entry["annotations_added"] > 0:
            entry["avg_seconds_per_annotation"] = round(
                entry["session_seconds"] / entry["annotations_added"], 2)
        else:
            entry["avg_seconds_per_annotation"] = 0.0

        # Remove image entry if it ended up with nothing
        if count == 0 and entry["annotations_added"] == 0:
            self._session_images.pop(img_name, None)
            self._session_annotated_images.discard(img_name)

        # Update persistent image_status
        if self._stats.get("image_status", {}).get(img_name) != "complete":
            self._stats["image_status"][img_name] = (
                "partial" if count > 0 else "unannotated")

    def _end_session(self):
        total_annotations = self._session_total_adds
        # Sum time only from images where annotations were added
        total_time = round(sum(
            img["session_seconds"] for img in self._session_images.values()
            if img.get("annotations_added", 0) > 0), 2)
        avg_time = (round(total_time / total_annotations, 2)
                    if total_annotations > 0 else 0.0)

        self._stats["sessions"].insert(0, {
            "user": self._current_user,
            "folder": self.image_folder or "",
            "started": self._session_start,
            "ended": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "images_annotated": len(self._session_annotated_images),
            "total_annotations": total_annotations,
            "total_time_seconds": total_time,
            "avg_seconds_per_annotation": avg_time,
            "images": {k: v for k, v in self._session_images.items()
                       if v.get("annotations_added", 0) > 0},
        })

    def _start_timer_display(self):
        self._update_timer_display()

    def _update_timer_display(self):
        if self._image_start_time and self.images:
            elapsed = time.time() - self._image_start_time
            mins, secs = divmod(int(elapsed), 60)
            self.status_time.configure(
                text=f"Image time: {mins}:{secs:02d}")
        self._timer_after_id = self.root.after(
            1000, self._update_timer_display)

    # ──────────────────────────────────────────────────────────────────────────
    #  Image status & filtering
    # ──────────────────────────────────────────────────────────────────────────
    def _load_completed_from_stats(self):
        """Populate _completed_images set from stats JSON on folder load."""
        self._completed_images = set()
        for img_name, status in self._stats.get("image_status", {}).items():
            if status == "complete":
                self._completed_images.add(img_name)

    def _get_image_status(self, img_name):
        """Return 'complete', 'partial', or 'unannotated' for an image."""
        return self._stats.get("image_status", {}).get(img_name, "unannotated")

    def _current_model_name(self):
        """Return the model name from the predictions manifest, or None if absent."""
        manifest = read_manifest(os.path.join(self.image_folder, "predictions"))
        return manifest.get("model") if manifest else None

    def _on_complete_toggled(self):
        """Complete is the dataset gate; it writes the completion record (spec 3.4)."""
        if not self.images:
            return
        img_name = self.images[self.index]
        store = self._stats_store
        if self._complete_var.get():
            blind = store.is_blind(img_name) and store.completion(img_name) is None
            store.set_completion(img_name, self._current_user, blind,
                                 len(self.document.annotations) if self.document else 0,
                                 None if blind else self._current_model_name())
            store.set_image_status(img_name, "complete")
            self._completed_images.add(img_name)
        else:
            store.clear_completion(img_name)
            self._completed_images.discard(img_name)
            store.set_image_status(
                img_name, "partial" if self._has_annotations(img_name) else "unannotated")
        self.save_current()
        self._rebuild_filter()
        self._update_filter_label()
        self._review_panel.load_predictions_for_current_image()
        self._review_panel.refresh(keep_focus=False)

    def _on_blind_toggled(self):
        """Handle the Blind checkbox toggle, hiding or restoring predictions."""
        if not self.images:
            return
        self._stats_store.set_blind(self.images[self.index], self._blind_var.get())
        self._save_stats()
        self._review_panel.load_predictions_for_current_image()
        self._review_panel.refresh(keep_focus=False)

    def _on_filter_changed(self, choice):
        """Handle filter dropdown selection."""
        mapping = {"All": "all", "Complete": "complete",
                   "Partial": "partial", "Unannotated": "unannotated"}
        self._active_filter = mapping.get(choice, "all")
        self._record_image_time()
        self._rebuild_filter()
        if self._filtered_indices:
            self.go_to_image(self._filtered_indices[0])
        else:
            self.save_current()
            # No images match the filter, so clear the canvas.
            self.original_image = None
            self.canvas.delete("all")
            self._annotate_tab._cached_scale = None
            self._annotate_tab._cached_tk_image = None
        self._update_filter_label()
        self.update_title()

    def _rebuild_filter(self):
        """Rebuild the list of image indices matching the active filter."""
        if self._active_filter == "all":
            self._filtered_indices = list(range(len(self.images)))
        else:
            self._filtered_indices = [
                i for i, name in enumerate(self.images)
                if self._get_image_status(name) == self._active_filter
            ]

    def _update_filter_label(self):
        """Update the total label to show filtered count when filtering."""
        if not self.images:
            return
        if self._active_filter == "all":
            self.total_label.configure(text=f"/ {len(self.images)}")
        else:
            self.total_label.configure(
                text=f"/ {len(self._filtered_indices)}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Class management
    # ──────────────────────────────────────────────────────────────────────────
    def _get_class_color(self, class_id):
        if class_id in self.class_colors:
            return self.class_colors[class_id]
        # Auto-assign from high-contrast palette and persist
        color = DEFAULT_CLASS_COLORS[class_id % len(DEFAULT_CLASS_COLORS)]
        self.class_colors[class_id] = color
        self._save_classes_file()
        return color

    def _count_class_annotations(self):
        """Count annotations per class for the current mode."""
        if self.document is None:
            return {}
        counts = {}
        for ann in self.document.annotations:
            if ann.kind != self.mode:
                continue
            counts[ann.class_id] = counts.get(ann.class_id, 0) + 1
        return counts

    def _refresh_class_dropdown(self):
        counts = self._count_class_annotations()
        items = []
        for cid, name in sorted(self.class_names.items()):
            c = counts.get(cid, 0)
            items.append(f"{cid}: {name} ({c})")
        items.append("<New Class>")
        self.class_dropdown.configure(values=items)
        active_count = counts.get(self.active_class, 0)
        active_label = (f"{self.active_class}: "
                        f"{self.class_names.get(self.active_class, '?')}"
                        f" ({active_count})")
        if active_label in items:
            self.class_dropdown.set(active_label)
        elif items:
            self.class_dropdown.set(items[0])

    def _on_class_selected(self, choice):
        if choice == "<New Class>":
            self._add_class_dialog()
            return
        try:
            class_id = int(choice.split(":")[0].strip())
        except ValueError:
            return
        self._select_class_by_id(class_id)

    def _class_name_dialog(self, text, title):
        """Build a class-name input dialog with the main window's icon; return the typed name or None."""
        dialog = ctk.CTkInputDialog(
            text=text, title=title, fg_color=BG_COLOR,
            button_fg_color=ACCENT, button_hover_color=ACCENT_HOVER,
            entry_fg_color=ENTRY_BG, entry_border_color=BORDER_COLOR,
            button_text_color=FG_COLOR)
        dialog.iconphoto(True, self._app_icon_image)
        if sys.platform.startswith("win"):
            dialog.iconbitmap(os.path.join(ASSETS_DIR, "app_icon.ico"))
        return dialog.get_input()

    def _add_class_dialog(self):
        """Open a small dialog to add a new class by name."""
        name = self._class_name_dialog("Enter new class name:", "Add Class")
        if not name or not name.strip():
            return
        name = name.strip()
        # Check if class already exists
        for cid, cname in self.class_names.items():
            if cname.lower() == name.lower():
                self.active_class = cid
                self._refresh_class_dropdown()
                self._update_color_btn()
                self.update_title()
                return
        next_id = (max(self.class_names.keys()) + 1
                   if self.class_names else 0)
        self.class_names[next_id] = name
        self.active_class = next_id
        print(f"[YoloLabeler] New class added: {next_id}: {name}")
        self._refresh_class_dropdown()
        self._update_color_btn()
        self._save_classes_file()
        self.update_title()

    def _rename_class_dialog(self):
        """Open a small dialog to rename the active class; a name another class holds is refused."""
        if self.active_class not in self.class_names:
            return
        current = self.class_names[self.active_class]
        name = self._class_name_dialog(
            f'Rename class {self.active_class} (currently "{current}"):',
            "Rename Class")
        if not name or not name.strip():
            return
        name = name.strip()
        for cid, cname in self.class_names.items():
            if cid != self.active_class and cname.lower() == name.lower():
                self.show_banner(
                    f'Class {cid} is already named "{cname}". '
                    f"Class {self.active_class} was not renamed.")
                return
        self.class_names[self.active_class] = name
        print(f"[YoloLabeler] Class renamed: {self.active_class}: "
              f"{current} -> {name}")
        self._refresh_class_dropdown()
        self._save_classes_file()
        self.update_title()

    def _select_class_by_id(self, class_id):
        if class_id in self.class_names:
            self.active_class = class_id
            self._refresh_class_dropdown()
            self._update_color_btn()
            self.update_title()
            if self.original_image is not None:
                self._annotate_tab.display_image()

    def _save_classes_file(self):
        """Save classes and colors to classes.json (unified format)."""
        # state_dir, not image_folder: the toolbar asks for a class colour before
        # _init_folder sets it, and an empty state_dir would write into the cwd.
        if not self.state_dir:
            return
        classes_path = os.path.join(self.state_dir, "classes.json")
        data = {}
        for cid in sorted(set(self.class_names) | set(self.class_colors)):
            entry = {}
            if cid in self.class_names:
                entry["name"] = self.class_names[cid]
            if cid in self.class_colors:
                entry["color"] = self.class_colors[cid]
            data[str(cid)] = entry
        try:
            with open(classes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            self.show_banner(f"Could not save classes.json: {e.strerror or e}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Class colors
    # ──────────────────────────────────────────────────────────────────────────
    def _update_color_btn(self):
        color = self._get_class_color(self.active_class)
        self.color_btn.config(bg=color, activebackground=color)

    def _pick_class_color(self):
        current = self._get_class_color(self.active_class)
        title = (f"Color for class {self.active_class} "
                 f"({self.class_names.get(self.active_class, '?')})")
        self._show_dark_color_picker(current, title)

    def _show_dark_color_picker(self, initial_color, title):
        """Custom dark-themed color picker with SI palette as custom colors."""
        picker = ctk.CTkToplevel(self.root)
        picker.title(title)
        picker.configure(fg_color=BG_COLOR)
        picker.geometry("360x440")
        picker.resizable(False, False)
        picker.transient(self.root)
        picker.grab_set()

        self._picker_result = None

        # SI custom palette
        si_palette = [
            ("SI Green", SI_GREEN), ("Water Blue", SI_WATER_BLUE),
            ("Mulberry", SI_MULBERRY), ("Lake Blue", SI_LAKE_BLUE),
            ("Stem Green", SI_STEM_GREEN), ("Persimmon", SI_PERSIMMON),
            ("Elderberry", SI_ELDERBERRY), ("Wood", SI_WOOD),
            ("Sage", SI_SAGE), ("Leaf Green", SI_LEAF_GREEN),
        ]

        # Basic palette
        basic_palette = [
            ("Red", "#FF0000"), ("Orange", "#FF8C00"),
            ("Yellow", "#FFD700"), ("Lime", "#32CD32"),
            ("Cyan", "#00CED1"), ("Blue", "#4169E1"),
            ("Purple", "#8A2BE2"), ("Pink", "#FF69B4"),
            ("White", "#FFFFFF"), ("Gray", "#808080"),
        ]

        # Preview swatch
        preview_var = tk.StringVar(value=initial_color)
        preview_frame = ctk.CTkFrame(picker, fg_color="transparent")
        preview_frame.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(preview_frame, text="Selected:",
                     font=(self.font_family, 12),
                     text_color=FG_COLOR).pack(side="left", padx=(0, 8))
        preview_swatch = tk.Label(preview_frame, text="    ", width=6,
                                  bg=initial_color, relief="solid", bd=1)
        preview_swatch.pack(side="left", padx=(0, 8))
        hex_entry = ctk.CTkEntry(preview_frame, width=90,
                                 font=(self.font_family, 11),
                                 fg_color=ENTRY_BG, border_color=BORDER_COLOR,
                                 text_color=FG_COLOR)
        hex_entry.pack(side="left")
        hex_entry.insert(0, initial_color)

        def _update_preview(color):
            preview_swatch.config(bg=color)
            hex_entry.delete(0, "end")
            hex_entry.insert(0, color)
            preview_var.set(color)

        def _on_hex_enter(event=None):
            val = hex_entry.get().strip()
            valid = False
            if len(val) == 7 and val.startswith("#"):
                try:
                    int(val[1:], 16)
                    valid = True
                except ValueError:
                    pass
            if valid:
                _update_preview(val)
            else:
                # Revert to last valid color
                _update_preview(preview_var.get())

        hex_entry.bind("<Return>", _on_hex_enter)

        def _make_swatch_grid(parent, palette):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            for i, (name, color) in enumerate(palette):
                btn = tk.Button(
                    frame, bg=color, activebackground=color,
                    width=3, height=1, relief="flat", bd=1,
                    command=lambda c=color: _update_preview(c))
                btn.grid(row=i // 5, column=i % 5, padx=3, pady=3)
            return frame

        # SI Colors section
        ctk.CTkLabel(picker, text="SI Palette",
                     font=(self.font_family, 11, "bold"),
                     text_color=FG_COLOR).pack(anchor="w", padx=12, pady=(6, 2))
        _make_swatch_grid(picker, si_palette).pack(padx=12, anchor="w")

        # Basic colors section
        ctk.CTkLabel(picker, text="Basic Colors",
                     font=(self.font_family, 11, "bold"),
                     text_color=FG_COLOR).pack(anchor="w", padx=12, pady=(10, 2))
        _make_swatch_grid(picker, basic_palette).pack(padx=12, anchor="w")

        # More... button to open system picker
        def _open_system_picker():
            result = colorchooser.askcolor(
                color=preview_var.get(),
                title="Choose custom color")
            if result and result[1]:
                _update_preview(result[1])

        ctk.CTkButton(
            picker, text="More Colors...", width=120,
            fg_color=ENTRY_BG, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=_open_system_picker
        ).pack(pady=(10, 6))

        # OK / Cancel
        btn_frame = ctk.CTkFrame(picker, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(4, 10))

        def _ok():
            self._picker_result = preview_var.get()
            picker.destroy()

        def _cancel():
            self._picker_result = None
            picker.destroy()

        ctk.CTkButton(
            btn_frame, text="OK", width=80,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12),
            command=_ok).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_frame, text="Cancel", width=80,
            fg_color=ENTRY_BG, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12),
            command=_cancel).pack(side="left")

        picker.wait_window()

        if self._picker_result:
            self.class_colors[self.active_class] = self._picker_result
            self._update_color_btn()
            self._save_classes_file()
            self._annotate_tab.display_image()
            self.canvas.update_idletasks()

    def _load_classes_json(self):
        """Load classes.json into the registry; returns the quarantined path if corrupt."""
        if not self.state_dir:
            return None
        data, moved = read_json_or_quarantine(
            os.path.join(self.state_dir, "classes.json"))
        if not isinstance(data, dict):
            return moved
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            try:
                cid = int(key)
            except ValueError:
                continue
            if "name" in entry:
                self.class_names[cid] = entry["name"]
            if "color" in entry:
                self.class_colors[cid] = entry["color"]
        return moved

    # ──────────────────────────────────────────────────────────────────────────
    #  Title & counter
    # ──────────────────────────────────────────────────────────────────────────
    def update_title(self):
        if not self.images:
            return
        self.root.title("YoloLabeler")
        # Show filtered position / filtered total
        if self._active_filter != "all":
            if self._filtered_indices:
                try:
                    filt_pos = self._filtered_indices.index(self.index) + 1
                except ValueError:
                    filt_pos = "-"
            else:
                filt_pos = 0
            self.counter_entry.delete(0, "end")
            self.counter_entry.insert(0, str(filt_pos))
            self.total_label.configure(
                text=f"/ {len(self._filtered_indices)}")
            if not self._filtered_indices:
                self.image_name_label.configure(text="No matches")
                self._complete_var.set(False)
                return
        else:
            self.counter_entry.delete(0, "end")
            self.counter_entry.insert(0, str(self.index + 1))
            self.total_label.configure(text=f"/ {len(self.images)}")
        self.image_name_label.configure(text=self.images[self.index])
        # Update complete checkbox to reflect current image
        img_name = self.images[self.index]
        self._complete_var.set(img_name in self._completed_images)
        self._blind_var.set(self._stats_store.is_blind(img_name))
        self._refresh_class_dropdown()

    # ──────────────────────────────────────────────────────────────────────────
    #  Editable image counter
    # ──────────────────────────────────────────────────────────────────────────
    def _on_counter_enter(self, event=None):
        text = self.counter_entry.get().strip()
        if not text:
            return
        try:
            num = int(text)
        except ValueError:
            self._on_counter_focus_out()
            return
        # When filtering, interpret as filtered position
        if self._active_filter != "all" and self._filtered_indices:
            if num < 1 or num > len(self._filtered_indices):
                self._on_counter_focus_out()
                return
            idx = self._filtered_indices[num - 1]
        else:
            idx = num - 1
        if idx < 0 or idx >= len(self.images):
            self._on_counter_focus_out()
            return
        self.go_to_image(idx)
        self.canvas.focus_set()

    def _on_counter_focus_out(self, event=None):
        if self.images:
            self.counter_entry.delete(0, "end")
            if self._active_filter != "all" and self._filtered_indices:
                try:
                    filt_pos = self._filtered_indices.index(self.index) + 1
                except ValueError:
                    filt_pos = "-"
                self.counter_entry.insert(0, str(filt_pos))
            else:
                self.counter_entry.insert(0, str(self.index + 1))

    # ──────────────────────────────────────────────────────────────────────────
    #  Shared helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _register_class_ids(self, class_ids):
        """Add placeholder names for class ids seen in files but not in classes.json."""
        unknown = [cid for cid in class_ids if cid not in self.class_names]
        if not unknown:
            return
        for cid in unknown:
            self.class_names[cid] = f"class_{cid}"
        self._refresh_class_dropdown()
        self._review_panel.refresh_class_filter()
        self._save_classes_file()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point for the yololabeler command."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.geometry("1200x800")
    root.title("YoloLabeler")
    root.configure(fg_color=BG_COLOR)

    folder = None
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        if not os.path.isdir(folder):
            print(f"Error: Invalid folder: {folder}")
            sys.exit(1)

    app = YoloLabeler(root, image_folder=folder)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
