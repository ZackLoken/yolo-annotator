"""
YoloLabeler v3 — Image Annotation & Review Tool for YOLO Training

Features:
- Draw bounding boxes (detection) or polygons (instance segmentation)
- Toggle between Box and Polygon mode via toolbar button or 'm' key
- Save annotations in YOLO format — labels/detect/ and labels/segment/ dirs
- Multi-class via editable dropdown (type new name + Enter to create)
- Custom class colors via color picker
- Ctrl+Scroll to zoom, Scroll to pan, Shift+Scroll to pan horizontally
- Middle-click drag to pan
- Undo/Redo (Ctrl+Z / Ctrl+Y)
- Right-click to delete annotations
- Full vertex editing in polygon mode (drag, insert on edge, delete)
- Vertex streaming: 'v' toggles stream mode; click starts/stops streaming
- Vertex snapping to existing polygon vertices (toggle 's', 5px radius)
- Editable image counter — type index + Enter to jump
- Help overlay grouped by Keyboard / Mouse sections (toggle 'h')
- Per-image time tracking with 5s threshold (JSON, tied to OS username)
- Auto EXIF orientation correction
- Boundary clamping for annotations
- Error handling for corrupt/unreadable images
- Auto-resume at last labeled image
- Cached image dimensions for fast CSV export on exit
- Savanna Institute branding: Archivo font, SI logo, SI accent color
"""

import os
import sys
import json
import time
import getpass
import datetime
from collections import namedtuple
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk

import customtkinter as ctk
import shutil

from yololabeler.state import AppState
from yololabeler.annotation.engine import AnnotationEngine
from yololabeler.annotation.tab import AnnotateTab
from yololabeler.review.engine import ReviewEngine
from yololabeler.review.tab import ReviewTab
from yololabeler.label_io import (
    parse_detect_predictions, parse_segment_predictions,
)
from yololabeler.matching import (
    compute_matches,
)
from yololabeler.utils import (
    suppress_tk_mac_warnings, _load_custom_fonts, _get_font_family,
    ASSETS_DIR,
)

# ── Constants ──────────────────────────────────────────────────────────────────
VERTEX_HANDLE_RADIUS = 4
STREAM_MIN_DISTANCE = 6   # min image-pixel distance between streamed vertices
SNAP_RADIUS = 15           # canvas-pixel radius for vertex/edge snapping

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

# ── Fixed review thresholds (match model.predict settings) ─────────────────────
REVIEW_IOU_THRESHOLD = 0.60
REVIEW_CONF_THRESHOLD = 0.50

# SI Brand colors — used for annotation class colors only
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

SI_CLASS_COLORS = [
    SI_GREEN, SI_WATER_BLUE, SI_MULBERRY, SI_LAKE_BLUE, SI_STEM_GREEN,
    SI_PERSIMMON, SI_ELDERBERRY, SI_WOOD, SI_SAGE, SI_LEAF_GREEN,
]

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


# ── Utilities ──────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  YoloLabeler  (v3)
# ══════════════════════════════════════════════════════════════════════════════

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
        'boxes', 'polygons', 'current_polygon', 'mode',
        'start_x', 'start_y', 'rect',
        # Predictions
        'pred_boxes', 'pred_polygons',
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
        '_selected_polygon_idx', '_hovered_polygon_idx',
        '_stream_mode', '_stream_active', '_last_stream_pos',
        # Review data
        '_review_index', '_review_detection_idx', '_review_detections',
        '_review_matches',
        '_review_gt_boxes', '_review_gt_polygons',
        '_review_pred_boxes', '_review_pred_polygons',
        '_review_original_image', '_review_img_w', '_review_img_h',
        '_review_scale', '_review_offset_x', '_review_offset_y',
        '_review_cached_scale',
        '_review_filter_type', '_review_filter_class',
        '_review_pan_start_x', '_review_pan_start_y',
        '_review_state', '_reviewed_lookup',
        '_review_show_gt', '_review_show_pred',
        '_review_filtered_images', '_review_status_filter',
        '_review_needs_first_zoom',
        '_review_det_reviewed', '_review_show_help',
        '_annotation_visible',
        '_annotate_pred_reference', '_review_return_pending',
        '_review_editing_det', '_review_recompute_on_return',
        # Stats & session
        '_stats', '_image_dims',
        '_current_user', '_session_start',
        '_image_start_time', '_review_image_start_time',
        '_session_annotated_images', '_session_images',
        '_session_loaded_counts', '_session_add_counts', '_session_total_adds',
        # Misc state
        '_defer_display', 'show_help',
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
        object.__setattr__(self, '_state', AppState())
        object.__setattr__(self, '_engine', AnnotationEngine(self._state))
        object.__setattr__(self, '_review', ReviewEngine(self._state))
        self.image_folder = image_folder or ""
        self.class_names = dict(class_names) if class_names else {}
        self.class_colors = {}
        self.images = []
        self.labels_dir = ""
        self.detect_dir = ""
        self.segment_dir = ""
        self.state_dir = ""
        self.img_width = 0
        self.img_height = 0
        self.original_image = None

        self.mode = "polygon"

        # State — boxes
        self.boxes = []
        self.start_x = None
        self.start_y = None
        self.rect = None

        # State — polygons
        self.polygons = []
        self.current_polygon = []
        self._dragging_vertex = None
        self._drag_orig_pos = None
        self._poly_bboxes = []
        self._poly_bboxes_dirty = True

        # State — predictions (review tab, read-only)
        self.pred_boxes = []
        self.pred_polygons = []
        self.pred_detect_dir = None
        self.pred_segment_dir = None

        # Review tab state
        self._review_index = 0
        self._review_detection_idx = 0
        self._review_detections = []
        self._review_matches = {}
        self._review_gt_boxes = []
        self._review_gt_polygons = []
        self._review_pred_boxes = []
        self._review_pred_polygons = []
        self._review_original_image = None
        self._review_img_w = 0
        self._review_img_h = 0
        self._review_scale = 1.0
        self._review_offset_x = 0.0
        self._review_offset_y = 0.0
        self._review_cached_scale = None
        self._review_cached_tk_image = None
        self._review_filter_type = "all"
        self._review_filter_class = "all"
        self._review_pan_start_x = None
        self._review_pan_start_y = None
        self._review_state = {}  # persisted review state
        self._reviewed_lookup = ("", {}, {})  # (img_name, pred_map, gt_map)
        self._review_show_gt = True
        self._review_show_pred = True
        self._review_filtered_images = []  # indices of images with preds/annotations
        self._review_status_filter = "all"  # all / not_reviewed / reviewed
        self._review_needs_first_zoom = False  # zoom on first Review tab switch
        self._review_det_reviewed = {}  # {img_name: set of reviewed det keys}
        self._annotation_visible = True
        self._review_show_help = False

        # Review → Annotate transition state
        self._annotate_pred_reference = None  # prediction overlay for Annotate
        self._review_return_pending = False   # True when editing from Review

        # Vertex streaming =
        self._stream_mode = False       # toggled by 'v' key
        self._stream_active = False     # currently recording stream
        self._last_stream_pos = None

        # Polygon hover tracking (for showing vertices on hover)
        self._hovered_polygon_idx = None

        # Explicit polygon selection for editing
        self._selected_polygon_idx = None

        # Vertex snapping
        self.snap_enabled = False

        # Undo / redo
        self._undo_stack = []       # snapshots of (boxes, polygons, selected_idx)
        self._redo_stack = []       # snapshots for redo
        self._vertex_redo_stack = [] # vertex-level redo while drawing

        # Common state
        self.active_class = 0
        self.show_help = False

        self._review_resize_after_id = None
        self.index = 0

        # Time tracking
        self._image_start_time = None
        self._review_image_start_time = None
        self._stats = {"sessions": [], "image_status": {}}
        self._current_user = getpass.getuser()
        self._session_start = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self._timer_after_id = None
        self._session_annotated_images = set()
        self._session_images = {}  # {img_name: per-image stats for current session}
        self._session_loaded_counts = {}  # {img_name: annotation count when loaded from disk}
        self._session_add_counts = {}  # {img_name: gross annotations added this session}
        self._session_total_adds = 0

        # Cached image dims for fast CSV export
        self._image_dims = {}

        # Completion tracking and filtering
        self._completed_images = set()
        self._active_filter = "all"  # "all", "complete", "partial", "unannotated"
        self._filtered_indices = []  # indices into self.images matching filter

        # Deferred display / review flags
        self._defer_display = False
        self._review_recompute_on_return = False
        self._review_editing_det = None

        # SI logo image ref (prevent GC)
        self._logo_image = None

        # ── Build GUI ──
        _load_custom_fonts()
        self.font_family = _get_font_family()
        self._build_toolbar()
        self._build_status_bar()

        # ── Tab view ──
        self.tabview = ctk.CTkTabview(
            self.root, fg_color=BG_COLOR, corner_radius=0,
            segmented_button_fg_color=BG_COLOR,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER,
            segmented_button_unselected_color=ENTRY_BG,
            segmented_button_unselected_hover_color=BORDER_COLOR,
            text_color=FG_COLOR,
            command=self._on_tab_changed)
        self.tabview.pack(fill="both", expand=True)
        self.tabview.add("Annotate")
        self.tabview.add("Review")

        # ── Annotate tab ──
        object.__setattr__(self, '_annotate_tab', AnnotateTab(self))
        self._annotate_tab.build(self.tabview.tab("Annotate"))
        self.canvas = self._annotate_tab.canvas  # backward compat

        # ── Review tab ──
        self._review_frame = self.tabview.tab("Review")
        object.__setattr__(self, '_review_tab', ReviewTab(self))
        self._review_tab.build()

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

        # ── CENTER: Visible | Class Dropdown | Mode | Stream | Snap ──
        self._toolbar_center = ctk.CTkFrame(inner, fg_color="transparent")
        self._toolbar_center.pack(side="left", padx=(8, 0))

        self._visible_var = tk.BooleanVar(value=True)
        self._visible_cb = ctk.CTkCheckBox(
            self._toolbar_center, text="Visible",
            variable=self._visible_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR,
            command=self._on_visible_toggled)
        self._visible_cb.pack(side="left", padx=(0, 4))

        self._toolbar_sep(self._toolbar_center)

        self.class_var = tk.StringVar()
        self.class_dropdown = ctk.CTkComboBox(
            self._toolbar_center, variable=self.class_var, width=180,
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

        self.color_btn = tk.Button(
            self._toolbar_center, text="  ", width=2, relief="flat",
            borderwidth=1, command=self._pick_class_color,
            bg=self._get_class_color(self.active_class),
            activebackground=self._get_class_color(self.active_class))
        self.color_btn.pack(side="left", padx=(2, 4))

        self._toolbar_sep(self._toolbar_center)

        self.mode_btn = ctk.CTkButton(
            self._toolbar_center, text="Mode: Polygon \u2b21", width=120,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_mode)
        self.mode_btn.pack(side="left", padx=(0, 4))

        self._toolbar_sep(self._toolbar_center)

        self.stream_btn = ctk.CTkButton(
            self._toolbar_center, text="Stream: Off", width=95,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_stream)
        self.stream_btn.pack(side="left", padx=(0, 4))

        self.snap_btn = ctk.CTkButton(
            self._toolbar_center, text="Snap: Off", width=80,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_snap)
        self.snap_btn.pack(side="left", padx=(0, 4))

        # ── RIGHT: always visible nav + annotate-only filter/complete ──
        self._toolbar_right = ctk.CTkFrame(inner, fg_color="transparent")
        self._toolbar_right.pack(side="right")

        self.next_btn = ctk.CTkButton(
            self._toolbar_right, text="Next \u25b6", width=70,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12, "bold"),
            command=self._nav_next)
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
            command=self._nav_prev)
        self.prev_btn.pack(side="right", padx=(4, 2))

        self.image_name_label = ctk.CTkLabel(
            self._toolbar_right, text="", font=(self.font_family, 11),
            text_color="#AAAAAA")
        self.image_name_label.pack(side="right", padx=(4, 2))

        # ── Annotate-only right widgets (hidden on Review tab) ──
        self._toolbar_annotate_right = ctk.CTkFrame(
            self._toolbar_right, fg_color="transparent")
        self._toolbar_annotate_right.pack(side="right")

        self._toolbar_sep_r(self._toolbar_annotate_right)

        self._complete_var = tk.BooleanVar(value=False)
        self.complete_cb = ctk.CTkCheckBox(
            self._toolbar_annotate_right, text="Complete",
            variable=self._complete_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR,
            command=self._on_complete_toggled)
        self.complete_cb.pack(side="right", padx=(4, 4))

        self._toolbar_sep_r(self._toolbar_annotate_right)

        self.filter_var = tk.StringVar(value="All")
        self.filter_dropdown = ctk.CTkComboBox(
            self._toolbar_annotate_right, variable=self.filter_var, width=130,
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
        self.filter_dropdown.pack(side="right", padx=(0, 4))

        ctk.CTkLabel(self._toolbar_annotate_right, text="Status:",
                     font=(self.font_family, 11),
                     text_color=FG_COLOR).pack(side="right", padx=(4, 2))

    def _toolbar_sep(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=28,
                           fg_color=BORDER_COLOR)
        sep.pack(side="left", padx=6, fill="y")

    def _toolbar_sep_r(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=28,
                           fg_color=BORDER_COLOR)
        sep.pack(side="right", padx=6, fill="y")

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

        # ── Review-only controls ──
        self._review_status_frame = ctk.CTkFrame(si, fg_color="transparent")
        # Not packed initially — shown when Review tab is active

        # LEFT group: Class | Type | Filter | DetLeft | Det# | DetRight
        ctk.CTkLabel(self._review_status_frame, text="Class:",
                     font=(self.font_family, 11),
                     text_color=FG_COLOR).pack(side="left", padx=(0, 2))
        self._review_class_var = tk.StringVar(value="All")
        self._review_class_dd = ctk.CTkComboBox(
            self._review_status_frame, variable=self._review_class_var,
            width=90,
            values=["All"],
            font=(self.font_family, 11),
            dropdown_font=(self.font_family, 11),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, dropdown_fg_color=BG_COLOR,
            dropdown_text_color=FG_COLOR, dropdown_hover_color=ACCENT,
            state="readonly",
            command=lambda c: self._review_tab._on_review_class_changed(c))
        self._review_class_dd.pack(side="left", padx=(0, 4))

        self._status_sep(self._review_status_frame)

        ctk.CTkLabel(self._review_status_frame, text="Type:",
                     font=(self.font_family, 11),
                     text_color=FG_COLOR).pack(side="left", padx=(0, 2))
        self._review_type_var = tk.StringVar(value="All")
        self._review_type_dd = ctk.CTkComboBox(
            self._review_status_frame, variable=self._review_type_var,
            width=80,
            values=["All", "FP", "FN", "TP"],
            font=(self.font_family, 11),
            dropdown_font=(self.font_family, 11),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, dropdown_fg_color=BG_COLOR,
            dropdown_text_color=FG_COLOR, dropdown_hover_color=ACCENT,
            state="readonly",
            command=lambda c: self._review_tab._on_review_type_changed(c))
        self._review_type_dd.pack(side="left", padx=(0, 4))

        self._status_sep(self._review_status_frame)

        ctk.CTkLabel(
            self._review_status_frame, text="Status:",
            font=(self.font_family, 11), text_color=FG_COLOR
        ).pack(side="left", padx=(0, 2))
        self._review_filter_var = tk.StringVar(value="All")
        self._review_filter_combo = ctk.CTkComboBox(
            self._review_status_frame, width=110,
            values=["All", "Not Reviewed", "Reviewed"],
            variable=self._review_filter_var,
            command=lambda c: self._review_tab._on_review_filter_changed(c),
            font=(self.font_family, 11),
            dropdown_font=(self.font_family, 11),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, dropdown_fg_color=BG_COLOR,
            dropdown_text_color=FG_COLOR, dropdown_hover_color=ACCENT,
            state="readonly")
        self._review_filter_combo.pack(side="left", padx=(0, 4))

        self._status_sep(self._review_status_frame)

        # Per-detection review status indicator (inside det nav group)
        self._review_det_status_label = ctk.CTkLabel(
            self._review_status_frame, text="",
            font=(self.font_family, 11), text_color=FG_COLOR,
            width=100, anchor="w")
        self._review_det_status_label.pack(side="left", padx=(0, 4))

        self._review_prev_det_btn = ctk.CTkButton(
            self._review_status_frame, text="\u25c0", width=30,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=lambda: self._review_tab._review_prev_detection())
        self._review_prev_det_btn.pack(side="left", padx=(0, 2))

        self._review_det_label = ctk.CTkLabel(
            self._review_status_frame, text="0 / 0",
            font=(self.font_family, 11), text_color=FG_COLOR,
            width=70, anchor="center")
        self._review_det_label.pack(side="left", padx=(2, 2))

        self._review_next_det_btn = ctk.CTkButton(
            self._review_status_frame, text="\u25b6", width=30,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=lambda: self._review_tab._review_next_detection())
        self._review_next_det_btn.pack(side="left", padx=(2, 4))

        self._status_sep(self._review_status_frame)

        # CENTER group: GT | Pred | Accept | Edit | Reject
        self._review_gt_var = tk.BooleanVar(value=True)
        self._review_gt_cb = ctk.CTkCheckBox(
            self._review_status_frame, text="GT",
            variable=self._review_gt_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR, width=40,
            command=lambda: self._review_tab._on_review_gt_toggled())
        self._review_gt_cb.pack(side="left", padx=(0, 2))

        self._review_pred_var = tk.BooleanVar(value=True)
        self._review_pred_cb = ctk.CTkCheckBox(
            self._review_status_frame, text="Pred",
            variable=self._review_pred_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR, width=45,
            command=lambda: self._review_tab._on_review_pred_toggled())
        self._review_pred_cb.pack(side="left", padx=(0, 4))

        self._review_action_sep = ctk.CTkFrame(
            self._review_status_frame, width=1, height=20,
            fg_color=BORDER_COLOR)
        self._review_action_sep.pack(side="left", padx=12, fill="y")

        self._review_accept_btn = ctk.CTkButton(
            self._review_status_frame, text="Accept (A)", width=110,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11, "bold"),
            command=lambda: self._review_tab._review_accept())
        self._review_accept_btn.pack(side="left", padx=(0, 4))

        self._review_edit_btn = ctk.CTkButton(
            self._review_status_frame, text="Edit (E)", width=75,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11, "bold"),
            command=lambda: self._review_tab._review_edit())
        self._review_edit_btn.pack(side="left", padx=(0, 4))

        self._review_reject_btn = ctk.CTkButton(
            self._review_status_frame, text="Reject (R)", width=110,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11, "bold"),
            command=lambda: self._review_tab._review_reject())
        self._review_reject_btn.pack(side="left", padx=(0, 4))

        # ── Right side: Counts | Zoom | Time | User ──
        # Counts label lives in shared right area, shown only on Review tab
        self._review_counts_label = ctk.CTkLabel(
            si, text="TP: 0 | FP: 0 | FN: 0",
            font=(self.font_family, 11), text_color=FG_COLOR)
        # Not packed initially — managed in _on_tab_changed
        self._review_counts_sep = ctk.CTkFrame(
            si, width=1, height=20, fg_color=BORDER_COLOR)
        # Not packed initially — managed in _on_tab_changed

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

    def _status_sep(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=20,
                           fg_color=BORDER_COLOR)
        sep.pack(side="left", padx=6, fill="y")

    def _status_sep_right(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=20,
                           fg_color=BORDER_COLOR)
        sep.pack(side="right", padx=6, fill="y")

    def _update_status(self):
        if self.tabview.get() == "Review":
            pct = int(self._review_scale * 100)
        else:
            pct = int(self._annotate_tab.scale * 100)
        self.status_zoom.configure(text=f"Zoom: {pct}%")

    def _on_tab_changed(self):
        """Handle tab switching between Annotate and Review."""
        if self.tabview.get() == "Review":
            self._review_tab.activate()
        elif self.tabview.get() == "Annotate":
            # Record review time, restart annotate timer
            self._record_review_time()
            if self.images:
                self._image_start_time = time.time()
            # Show annotate toolbar sections, hide review controls
            self._review_status_frame.pack_forget()
            self._review_counts_label.pack_forget()
            self._review_counts_sep.pack_forget()
            self._toolbar_center.pack(side="left", padx=(8, 0))
            self._toolbar_annotate_right.pack(side="right")
            # Sync viewport from review → annotate
            at = self._annotate_tab
            if self._review_original_image is not None and self.images:
                # Navigate to same image as review
                if self.index != self._review_index:
                    self._defer_display = True
                    self.index = self._review_index
                    at.load_image()
                    self._defer_display = False
                # Copy review zoom/offset to annotate
                at.scale = self._review_scale
                at.offset_x = self._review_offset_x
                at.offset_y = self._review_offset_y
                at.zoom_index = at._nearest_zoom_index(self._review_scale)
                at._cached_scale = None
            # Default to polygon mode if polygon labels exist
            if self.mode != "polygon" and self.polygons:
                self.mode = "polygon"
                self.mode_btn.configure(text="Mode: Polygon \u2b21")
                self.stream_btn.pack(side="left", padx=(0, 4))
                self.snap_btn.pack(side="left", padx=(0, 4))
            if self.original_image is not None:
                at.display_image()
            self.update_title()
            self._update_status()

    def _nav_prev(self):
        """Context-aware previous: image in Annotate, image in Review."""
        if self.tabview.get() == "Review":
            self._review_tab._review_prev_image()
        else:
            self._annotate_tab.prev_image()

    def _nav_next(self):
        """Context-aware next: image in Annotate, image in Review."""
        if self.tabview.get() == "Review":
            self._review_tab._review_next_image()
        else:
            self._annotate_tab.next_image()

    def _on_visible_toggled(self):
        """Toggle annotation visibility in Annotate tab."""
        self._annotation_visible = self._visible_var.get()
        if self.original_image is not None:
            self._annotate_tab.display_image()

    def _rebuild_review_image_list(self):
        """Build filtered list of image indices for Review tab."""
        self._review_tab._rebuild_review_image_list()

    # ──────────────────────────────────────────────────────────────────────────
    #  Logo
    # ──────────────────────────────────────────────────────────────────────────
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
    def _setup_bindings(self):
        """Bind keyboard shortcuts (canvas bindings are in AnnotateTab/ReviewTab)."""
        c = self.canvas

        r = self.root
        r.bind("<Right>", self._on_right_key)
        r.bind("<Left>", self._on_left_key)
        r.bind("<Up>", self._on_up_key)
        r.bind("<Down>", self._on_down_key)
        r.bind("<Escape>", self._on_escape)
        r.bind("<Control-z>", self._annotate_tab.undo_last)
        r.bind("<Control-y>", self._annotate_tab.redo_last)
        if sys.platform == "darwin":
            r.bind("<Command-z>", self._annotate_tab.undo_last)
            r.bind("<Command-y>", self._annotate_tab.redo_last)

        r.bind("h", lambda e: self._help_key())
        r.bind("m", lambda e: self._key_action(self._toggle_mode))
        r.bind("s", lambda e: self._annotate_key(self._toggle_snap))
        r.bind("v", lambda e: self._annotate_key(self._toggle_stream))

        r.bind("a", lambda e: self._review_key(self._review_tab._review_accept))
        r.bind("r", lambda e: self._review_key(self._review_tab._review_reject))
        r.bind("e", lambda e: self._review_key(self._review_tab._review_edit))
        r.bind("<space>", self._on_space_key)

        for key_num in range(10):
            r.bind(str(key_num),
                   lambda e, cid=key_num: self._key_action(
                       lambda: self._select_class_by_id(cid)))

        r.protocol("WM_DELETE_WINDOW", self._quit)
        c.focus_set()
        c.bind("<Enter>", lambda e: c.focus_set())

    def _key_action(self, action):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if focused is not None:
            parent = focused.master
            if isinstance(parent, ctk.CTkComboBox):
                return
        action()

    def _annotate_key(self, action):
        if self.tabview.get() != "Annotate":
            return
        self._key_action(action)

    def _review_key(self, action):
        if self.tabview.get() != "Review":
            return
        self._key_action(action)

    def _on_right_key(self, event=None):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if self.tabview.get() == "Review":
            self._review_tab._review_next_detection()
        else:
            self._annotate_tab.next_image()

    def _on_left_key(self, event=None):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if self.tabview.get() == "Review":
            self._review_tab._review_prev_detection()
        else:
            self._annotate_tab.prev_image()

    def _on_up_key(self, event=None):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if self.tabview.get() == "Review":
            self._review_tab._review_next_image()

    def _on_down_key(self, event=None):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if self.tabview.get() == "Review":
            self._review_tab._review_prev_image()

    def _help_key(self):
        """Toggle help in both tabs."""
        self._key_action(self._annotate_tab.toggle_help)

    def _on_space_key(self, event):
        """Spacebar = left click at current cursor position (annotate only)."""
        if self.tabview.get() != "Annotate":
            return
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if focused is not None:
            parent = focused.master
            if isinstance(parent, ctk.CTkComboBox):
                return
        # Get cursor position relative to the canvas
        cx = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        cy = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        # Create a synthetic event
        fake = _SynthEvent(cx, cy)
        self._annotate_tab.on_button_press(fake)

    # ──────────────────────────────────────────────────────────────────────────
    #  Mode toggle
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_mode(self, event=None):
        if self.mode == "box":
            self.mode = "polygon"
            self.mode_btn.configure(text="Mode: Polygon \u2b21")
            self.stream_btn.pack(side="left", padx=(0, 4))
            self.snap_btn.pack(side="left", padx=(0, 4))
        else:
            self.mode = "box"
            self.mode_btn.configure(text="Mode: Box \u25ad")
            self.current_polygon = []
            self._dragging_vertex = None
            self._drag_orig_pos = None
            self._selected_polygon_idx = None
            self._stream_mode = False
            self._stream_active = False
            self.stream_btn.configure(text="Stream: Off")
            self.stream_btn.pack_forget()
            self.snap_btn.pack_forget()
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
        self.images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
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

        # Prediction directories (for review tab)
        self.pred_detect_dir = os.path.join(folder, "predictions", "detect")
        self.pred_segment_dir = os.path.join(folder, "predictions", "segment")
        os.makedirs(self.pred_detect_dir, exist_ok=True)
        os.makedirs(self.pred_segment_dir, exist_ok=True)

        # Preserve classes added before folder was opened, then load JSON
        pre_open_names = dict(self.class_names)
        pre_open_colors = dict(self.class_colors)
        self._load_classes_json()
        # Merge: keep pre-open classes that weren't in the JSON
        for cid, name in pre_open_names.items():
            if cid not in self.class_names:
                self.class_names[cid] = name
        for cid, color in pre_open_colors.items():
            if cid not in self.class_colors:
                self.class_colors[cid] = color
        self._refresh_class_dropdown()

        self._load_stats()
        self._load_completed_from_stats()
        self._load_review_state()
        # Prepopulate image_status for every image in the folder
        for img_name in self.images:
            if img_name not in self._stats["image_status"]:
                if self._has_annotations(img_name):
                    self._stats["image_status"][img_name] = "partial"
                else:
                    self._stats["image_status"][img_name] = "unannotated"
        self._save_stats()
        self._image_dims = {}
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
                self.mode = "box"
                self.mode_btn.configure(text="Mode: Box \u25ad")
                self.stream_btn.pack_forget()
                self.snap_btn.pack_forget()

        # Persist classes (merges pre-open + JSON + ensures file exists)
        self._save_classes_file()

        # Pre-cache the review filtered image list so switching tabs is fast
        self._rebuild_review_image_list()

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
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 800
        self.canvas.create_text(
            cw // 2, ch // 2,
            text='Click "Open Folder" to load images',
            fill=FG_COLOR, font=(self.font_family, 16),
            tags="welcome")
        self.root.title("YoloLabeler")
        # Re-center on resize
        self._welcome_bind_id = self.canvas.bind(
            "<Configure>", self._reposition_welcome)
        # Show welcome on review canvas too
        rc = self._review_canvas
        rc.delete("all")
        rcw = rc.winfo_width() or 1200
        rch = rc.winfo_height() or 800
        rc.create_text(
            rcw // 2, rch // 2,
            text='Click "Open Folder" to load images',
            fill=FG_COLOR, font=(self.font_family, 16),
            tags="review_welcome")
        self._review_welcome_bind_id = rc.bind(
            "<Configure>", self._reposition_review_welcome)

    def _reposition_review_welcome(self, event=None):
        """Keep review welcome text centered when canvas resizes."""
        items = self._review_canvas.find_withtag("review_welcome")
        if items:
            cw = self._review_canvas.winfo_width()
            ch = self._review_canvas.winfo_height()
            self._review_canvas.coords(items[0], cw // 2, ch // 2)
        else:
            if hasattr(self, "_review_welcome_bind_id"):
                self._review_canvas.unbind(
                    "<Configure>", self._review_welcome_bind_id)
                del self._review_welcome_bind_id

    def _reposition_welcome(self, event=None):
        """Keep welcome text centered when canvas resizes."""
        items = self.canvas.find_withtag("welcome")
        if items:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            self.canvas.coords(items[0], cw // 2, ch // 2)
        else:
            # Welcome text gone (folder opened), unbind
            if hasattr(self, "_welcome_bind_id"):
                self.canvas.unbind("<Configure>", self._welcome_bind_id)
                del self._welcome_bind_id

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
            try:
                self._record_image_time()
                self._annotate_tab.save_annotations()
                self._end_session()
                self._save_stats()
            except Exception as e:
                print(f"Warning: Could not save before opening new folder: {e}")

        self._init_folder(new_folder)

        if not self.images:
            messagebox.showinfo("No images",
                                "No images found in the folder!")
            return

        self._session_start = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self._session_annotated_images = set()
        self._session_images = {}
        self._session_loaded_counts = {}
        self._session_add_counts = {}
        self._session_total_adds = 0
        self._annotate_tab.load_image()

        # Reset and pre-load review for the new folder
        self._review_original_image = None
        self._review_index = 0
        self._review_detection_idx = 0
        self._review_needs_first_zoom = False
        if self._review_filtered_images:
            self._review_index = self._review_filtered_images[0]
            self._review_tab._review_load_image()
            self._review_needs_first_zoom = True

    # ──────────────────────────────────────────────────────────────────────────
    #  Quit
    # ──────────────────────────────────────────────────────────────────────────
    def _quit(self):
        if self.image_folder and self.images:
            print("[YoloLabeler] Saving and closing...")
            try:
                self._record_image_time()
                self._record_review_time()
                self._annotate_tab.save_annotations()
                self._end_session()
                self._save_stats()
                print("[YoloLabeler] Done.")
            except Exception as e:
                print(f"Warning: Could not save on exit: {e}")
        if self._timer_after_id:
            self.root.after_cancel(self._timer_after_id)
        if self._annotate_tab._resize_after_id:
            self.root.after_cancel(self._annotate_tab._resize_after_id)
        if self._review_resize_after_id:
            self.root.after_cancel(self._review_resize_after_id)
        self.root.destroy()

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
            if self._selected_polygon_idx is not None:
                self._selected_polygon_idx = None
                self._annotate_tab._clear_drag_state()
                self._annotate_tab.display_image()
                if self._review_return_pending:
                    self.root.after(50, self._review_tab._review_confirm_dialog)

    # ──────────────────────────────────────────────────────────────────────────
    #  Time tracking
    # ──────────────────────────────────────────────────────────────────────────
    def _stats_path(self):
        if self.image_folder:
            return os.path.join(self.state_dir, "annotation_stats.json")
        return None

    def _load_stats(self):
        path = self._stats_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._stats = json.load(f)
                if "sessions" not in self._stats:
                    self._stats["sessions"] = []
                if "image_status" not in self._stats:
                    self._stats["image_status"] = {}
                # Migrate old format: extract completion from top-level "images"
                if "images" in self._stats:
                    for iname, ientry in self._stats["images"].items():
                        if ientry.get("status") == "complete":
                            self._stats["image_status"].setdefault(
                                iname, "complete")
                    del self._stats["images"]
            except Exception:
                self._stats = {"sessions": [], "image_status": {}}
        else:
            self._stats = {"sessions": [], "image_status": {}}

    def _save_stats(self):
        path = self._stats_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save stats: {e}")

    # ── Review state persistence ─────────────────────────────────────────────

    def _review_state_path(self):
        return self._review.review_state_path()

    def _load_review_state(self):
        self._review.load_review_state()

    def _save_review_state(self):
        self._review.save_review_state()

    def _mark_image_reviewed(self, img_name):
        self._review.mark_image_reviewed(img_name)

    def _is_image_reviewed(self, img_name):
        return self._review.is_image_reviewed(img_name)

    def _get_image_review_status(self, img_name):
        return self._review.get_image_review_status(img_name)

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

        count = len(self.boxes) + len(self.polygons)
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
        if self.tabview.get() == "Review":
            if self._review_image_start_time:
                elapsed = time.time() - self._review_image_start_time
                mins, secs = divmod(int(elapsed), 60)
                self.status_time.configure(
                    text=f"Review time: {mins}:{secs:02d}")
        else:
            if self._image_start_time and self.images:
                elapsed = time.time() - self._image_start_time
                mins, secs = divmod(int(elapsed), 60)
                self.status_time.configure(
                    text=f"Image time: {mins}:{secs:02d}")
        self._timer_after_id = self.root.after(
            1000, self._update_timer_display)

    def _record_review_time(self):
        """Record elapsed review time and reset the review timer."""
        if self._review_image_start_time is None:
            return
        elapsed = time.time() - self._review_image_start_time
        self._review_image_start_time = None
        if not self.images:
            return
        img_name = self.images[self._review_index]
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(
            img_name, {"img_status": "not_started", "detections": []})
        img_data["review_seconds"] = round(
            img_data.get("review_seconds", 0.0) + elapsed, 2)
        self._save_review_state()

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

    def _on_complete_toggled(self):
        """Handle the Complete checkbox toggle."""
        if not self.images:
            return
        img_name = self.images[self.index]
        if self._complete_var.get():
            self._completed_images.add(img_name)
            status = "complete"
        else:
            self._completed_images.discard(img_name)
            status = "partial" if self._has_annotations(img_name) else "unannotated"
        # Persist completion status
        self._stats["image_status"][img_name] = status
        self._save_stats()
        self._rebuild_filter()
        self._update_filter_label()

    def _on_filter_changed(self, choice):
        """Handle filter dropdown selection."""
        mapping = {"All": "all", "Complete": "complete",
                   "Partial": "partial", "Unannotated": "unannotated"}
        self._active_filter = mapping.get(choice, "all")
        self._record_image_time()
        self._annotate_tab.save_annotations()
        self._save_stats()
        self._rebuild_filter()
        if self._filtered_indices:
            self.index = self._filtered_indices[0]
            self._annotate_tab.load_image()
        else:
            # No images match filter — clear the canvas
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
        counts = {}
        if self.mode == "box":
            for *_, cls in self.boxes:
                counts[cls] = counts.get(cls, 0) + 1
        else:
            for _, cls in self.polygons:
                counts[cls] = counts.get(cls, 0) + 1
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
            self.active_class = class_id
            self._update_color_btn()
            self._refresh_class_dropdown()
            self.update_title()
            if self.original_image is not None:
                self._annotate_tab.display_image()
        except (ValueError, IndexError):
            pass

    def _add_class_dialog(self):
        """Open a small dialog to add a new class by name."""
        dialog = ctk.CTkInputDialog(
            text="Enter new class name:",
            title="Add Class",
            fg_color=BG_COLOR,
            button_fg_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            entry_fg_color=ENTRY_BG,
            entry_border_color=BORDER_COLOR,
            button_text_color=FG_COLOR)
        name = dialog.get_input()
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
        if not self.image_folder:
            return
        classes_path = os.path.join(self.state_dir, "classes.json")
        data = {}
        for cid in sorted(self.class_names.keys()):
            data[str(cid)] = {
                "name": self.class_names[cid],
            }
            if cid in self.class_colors:
                data[str(cid)]["color"] = self.class_colors[cid]
        try:
            with open(classes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"Warning: Could not save classes.json: {e}")

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
            if self.tabview.get() == "Review":
                self._review_tab._display_review_image()
            elif self.original_image is not None:
                self._annotate_tab.display_image()
                self.canvas.update_idletasks()

    def _load_classes_json(self):
        """Load classes and colors from classes.json."""
        if not self.image_folder:
            return
        classes_json_path = os.path.join(self.state_dir, "classes.json")
        if os.path.exists(classes_json_path):
            try:
                with open(classes_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.class_names = {}
                self.class_colors = {}
                for k, v in data.items():
                    cid = int(k)
                    self.class_names[cid] = v.get("name", f"class_{cid}")
                    if "color" in v:
                        self.class_colors[cid] = v["color"]
            except Exception:
                pass

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
        self._record_image_time()
        self._annotate_tab.save_annotations()
        self._save_stats()
        self.index = idx
        self._annotate_tab.load_image()
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
    #  Shared helpers (used by both Annotate and Review tabs)
    # ──────────────────────────────────────────────────────────────────────────
    def _load_predictions(self, image_name, img_w, img_h):
        """Load model predictions for an image."""
        pred_boxes = []
        pred_polygons = []
        if not self.pred_detect_dir or not self.pred_segment_dir:
            return pred_boxes, pred_polygons
        stem = os.path.splitext(image_name)[0]

        detect_path = os.path.join(self.pred_detect_dir, f"{stem}.txt")
        try:
            pboxes, det_cids = parse_detect_predictions(
                detect_path, img_w, img_h)
            pred_boxes.extend(pboxes)
            for cid in det_cids:
                if cid not in self.class_names:
                    self.class_names[cid] = f"class_{cid}"
                    self._refresh_class_dropdown()
                    self._save_classes_file()
        except Exception as e:
            print(f"Warning: Could not load detect predictions for {stem}: {e}")

        segment_path = os.path.join(self.pred_segment_dir, f"{stem}.txt")
        try:
            ppolys, seg_cids = parse_segment_predictions(
                segment_path, img_w, img_h)
            pred_polygons.extend(ppolys)
            for cid in seg_cids:
                if cid not in self.class_names:
                    self.class_names[cid] = f"class_{cid}"
                    self._refresh_class_dropdown()
                    self._save_classes_file()
        except Exception as e:
            print(f"Warning: Could not load segment predictions for {stem}: {e}")

        return pred_boxes, pred_polygons

    def _compute_matches(self, gt_boxes, gt_polygons, pred_boxes, pred_polygons,
                         iou_threshold=0.5, conf_threshold=0.25):
        """Match predictions to ground truth using IoU."""
        return compute_matches(gt_boxes, gt_polygons, pred_boxes,
                               pred_polygons, iou_threshold, conf_threshold)


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
