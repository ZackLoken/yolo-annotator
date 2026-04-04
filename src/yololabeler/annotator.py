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
import math
import time
import getpass
import datetime
import contextlib
import copy
from collections import namedtuple
import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ExifTags

import customtkinter as ctk
import shutil

from yololabeler.label_io import (
    parse_detect_labels, parse_segment_labels,
    parse_detect_predictions, parse_segment_predictions,
    write_detect_labels, write_segment_labels,
)
from yololabeler.matching import (
    point_to_segment_dist, point_in_polygon,
    box_iou, polygon_iou, box_to_points, compute_matches,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

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

@contextlib.contextmanager
def suppress_tk_mac_warnings():
    if sys.platform == "darwin":
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    else:
        yield


_CUSTOM_FONT_LOADED = False


def _load_custom_fonts():
    global _CUSTOM_FONT_LOADED
    if not os.path.isdir(ASSETS_DIR):
        return False
    if sys.platform.startswith("win"):
        try:
            import ctypes
            FR_PRIVATE = 0x10
            gdi32 = ctypes.windll.gdi32
            for name in ("Archivo-Regular.ttf", "Archivo-Bold.ttf",
                         "Archivo-Medium.ttf", "Archivo-SemiBold.ttf"):
                path = os.path.join(ASSETS_DIR, name)
                if os.path.exists(path):
                    gdi32.AddFontResourceExW(path, FR_PRIVATE, 0)
            _CUSTOM_FONT_LOADED = True
            return True
        except Exception:
            return False
    elif sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util
            ct_path = ctypes.util.find_library("CoreText")
            if ct_path:
                ct = ctypes.cdll.LoadLibrary(ct_path)
                cf_path = ctypes.util.find_library("CoreFoundation")
                cf = ctypes.cdll.LoadLibrary(cf_path)
                for name in ("Archivo-Regular.ttf", "Archivo-Bold.ttf",
                             "Archivo-Medium.ttf", "Archivo-SemiBold.ttf"):
                    path = os.path.join(ASSETS_DIR, name)
                    if os.path.exists(path):
                        url_ref = cf.CFURLCreateFromFileSystemRepresentation(
                            None, path.encode("utf-8"), len(path.encode("utf-8")), False)
                        if url_ref:
                            ct.CTFontManagerRegisterFontsForURL(url_ref, 1, None)
                _CUSTOM_FONT_LOADED = True
                return True
        except Exception:
            return False
    return False


def _get_font_family():
    if _CUSTOM_FONT_LOADED:
        return "Archivo"
    # Fallback to common cross-platform fonts
    try:
        available = set(tkFont.families())
    except Exception:
        available = set()
    for family in ("Segoe UI", "Helvetica Neue", "Helvetica",
                   "Arial", "DejaVu Sans", "sans-serif"):
        if family in available:
            return family
    return "TkDefaultFont"


def auto_orient_image(img):
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return img
        orientation = exif[orientation_key]
        ops = {
            2: lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            3: lambda i: i.rotate(180, expand=True),
            4: lambda i: i.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            5: lambda i: i.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT).rotate(270, expand=True),
            6: lambda i: i.rotate(270, expand=True),
            7: lambda i: i.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT).rotate(90, expand=True),
            8: lambda i: i.rotate(90, expand=True),
        }
        if orientation in ops:
            img = ops[orientation](img)
    except Exception as e:
        print(f"Warning: Could not auto-orient image: {e}")
    return img


# Backwards-compatible alias — callers use _point_to_segment_dist
_point_to_segment_dist = point_to_segment_dist


# ══════════════════════════════════════════════════════════════════════════════
#  YoloLabeler  (v3)
# ══════════════════════════════════════════════════════════════════════════════

class YoloLabeler:
    def __init__(self, root, image_folder=None, class_names=None):
        self.root = root
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
        self._poly_preview_line = None
        self._mouse_canvas_x = 0
        self._mouse_canvas_y = 0
        self._poly_bboxes = []          # precomputed (min_x, min_y, max_x, max_y) per polygon
        self._poly_bboxes_dirty = True  # rebuild on next spatial query

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
        self._snap_indicator_item = None

        # Undo / redo
        self._undo_stack = []       # snapshots of (boxes, polygons, selected_idx)
        self._redo_stack = []       # snapshots for redo
        self._vertex_redo_stack = [] # vertex-level redo while drawing

        # Common state
        self.active_class = 0
        self.show_help = False

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

        self._redraw_pending = False
        self._motion_last_time = 0.0
        self._resize_after_id = None
        self._review_resize_after_id = None
        self._fast_resample = False
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

        # ── Annotate canvas ──
        self.canvas = tk.Canvas(
            self.tabview.tab("Annotate"), cursor="cross",
            bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # ── Review tab ──
        self._review_frame = self.tabview.tab("Review")
        self._build_review_tab()

        self._setup_bindings()

        # ── Start ──
        if self.image_folder:
            self._init_folder(self.image_folder)
            self.root.after(100, self.load_image)
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
            command=self._on_review_class_changed)
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
            command=self._on_review_type_changed)
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
            command=self._on_review_filter_changed,
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
            font=(self.font_family, 11), text_color=FG_COLOR)
        self._review_det_status_label.pack(side="left", padx=(0, 4))

        self._review_prev_det_btn = ctk.CTkButton(
            self._review_status_frame, text="\u25c0", width=30,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._review_prev_detection)
        self._review_prev_det_btn.pack(side="left", padx=(0, 2))

        self._review_det_label = ctk.CTkLabel(
            self._review_status_frame, text="0 / 0",
            font=(self.font_family, 11), text_color=FG_COLOR)
        self._review_det_label.pack(side="left", padx=(2, 2))

        self._review_next_det_btn = ctk.CTkButton(
            self._review_status_frame, text="\u25b6", width=30,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._review_next_detection)
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
            command=self._on_review_gt_toggled)
        self._review_gt_cb.pack(side="left", padx=(0, 2))

        self._review_pred_var = tk.BooleanVar(value=True)
        self._review_pred_cb = ctk.CTkCheckBox(
            self._review_status_frame, text="Pred",
            variable=self._review_pred_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR, width=45,
            command=self._on_review_pred_toggled)
        self._review_pred_cb.pack(side="left", padx=(0, 4))

        self._review_action_sep = ctk.CTkFrame(
            self._review_status_frame, width=1, height=20,
            fg_color=BORDER_COLOR)
        self._review_action_sep.pack(side="left", padx=12, fill="y")

        self._review_accept_btn = ctk.CTkButton(
            self._review_status_frame, text="Accept (A)", width=90,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11, "bold"),
            command=self._review_accept)
        self._review_accept_btn.pack(side="left", padx=(0, 4))

        self._review_edit_btn = ctk.CTkButton(
            self._review_status_frame, text="Edit (E)", width=75,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11, "bold"),
            command=self._review_edit)
        self._review_edit_btn.pack(side="left", padx=(0, 4))

        self._review_reject_btn = ctk.CTkButton(
            self._review_status_frame, text="Reject (R)", width=90,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11, "bold"),
            command=self._review_reject)
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
            pct = int(self.scale * 100)
        self.status_zoom.configure(text=f"Zoom: {pct}%")

    def _on_tab_changed(self):
        """Handle tab switching between Annotate and Review."""
        if self.tabview.get() == "Review":
            # Record annotate time, start review time
            self._record_image_time()
            if self._review_image_start_time is None:
                self._review_image_start_time = time.time()
            # Back up original labels before any review actions
            self._backup_original_labels()
            # Clean up prediction reference from annotate
            if self._review_return_pending:
                self._annotate_pred_reference = None
                self._review_return_pending = False
            # Show review controls, hide annotate-only toolbar sections
            self._toolbar_center.pack_forget()
            self._toolbar_annotate_right.pack_forget()
            self._review_status_frame.pack(side="left")
            # Show counts in the right-side status area
            self._review_counts_sep.pack(side="right", padx=6, fill="y")
            self._review_counts_label.pack(side="right", padx=(6, 6))
            # Returning from edit: reload GT, recompute, and advance
            if getattr(self, '_review_recompute_on_return', False):
                self._review_recompute_on_return = False
                self._review_reload_gt_and_advance()
            # Returning from annotate (manual tab switch): recompute in
            # case the user added/modified GT annotations ad-hoc.
            elif (self._review_original_image is not None
                  and self._review_matches is not None):
                self._review_reload_gt_and_advance()
            # Normal switch: full image load or restore
            elif self.images:
                if self._review_original_image is None:
                    # First switch — pick the first reviewable image
                    if self._review_filtered_images:
                        self._review_index = self._review_filtered_images[0]
                    else:
                        # No images to review
                        self._display_review_image()
                        self._update_review_labels()
                        self._update_status()
                        return
                    self._review_load_image()
                    self._review_needs_first_zoom = True
                elif not self._review_filtered_images:
                    # Returning to review but nothing to review
                    self._review_original_image = None
                    self._display_review_image()
                    self._update_review_labels()
                    self._update_status()
                    return
                else:
                    # Already loaded — preserve det index and zoom
                    self._review_cached_scale = None
                    self._display_review_image()
                # Zoom to first detection on initial switch
                if self._review_needs_first_zoom:
                    self._review_needs_first_zoom = False
                    self._review_zoom_to_first_unreviewed()
            self._update_review_labels()
            self._update_status()
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
            if self._review_original_image is not None and self.images:
                # Navigate to same image as review
                if self.index != self._review_index:
                    self._defer_display = True
                    self.index = self._review_index
                    self.load_image()
                    self._defer_display = False
                # Copy review zoom/offset to annotate
                self.scale = self._review_scale
                self.offset_x = self._review_offset_x
                self.offset_y = self._review_offset_y
                self.zoom_index = self._nearest_zoom_index(self._review_scale)
                self._cached_scale = None
            # Default to polygon mode if polygon labels exist
            if self.mode != "polygon" and self.polygons:
                self.mode = "polygon"
                self.mode_btn.configure(text="Mode: Polygon \u2b21")
                self.stream_btn.pack(side="left", padx=(0, 4))
                self.snap_btn.pack(side="left", padx=(0, 4))
            if self.original_image is not None:
                self.display_image()
            self.update_title()
            self._update_status()

    def _nav_prev(self):
        """Context-aware previous: image in Annotate, image in Review."""
        if self.tabview.get() == "Review":
            self._review_prev_image()
        else:
            self.prev_image()

    def _nav_next(self):
        """Context-aware next: image in Annotate, image in Review."""
        if self.tabview.get() == "Review":
            self._review_next_image()
        else:
            self.next_image()

    def _on_visible_toggled(self):
        """Toggle annotation visibility in Annotate tab."""
        self._annotation_visible = self._visible_var.get()
        if self.original_image is not None:
            self.display_image()

    def _on_review_gt_toggled(self):
        """Toggle GT visibility in Review tab."""
        self._review_show_gt = self._review_gt_var.get()
        self._display_review_image()

    def _on_review_pred_toggled(self):
        """Toggle prediction visibility in Review tab."""
        self._review_show_pred = self._review_pred_var.get()
        self._display_review_image()

    def _rebuild_review_image_list(self):
        """Build filtered list of image indices for Review tab.

        Only includes images that have predictions and/or annotations.
        Images with neither are skipped entirely.
        """
        if not self.images:
            self._review_filtered_images = []
            return
        filtered = []
        has_any_preds = False
        for img_name in self.images:
            stem = os.path.splitext(img_name)[0]
            has_pred = (
                (self.pred_detect_dir and os.path.exists(
                    os.path.join(self.pred_detect_dir, f"{stem}.txt")))
                or (self.pred_segment_dir and os.path.exists(
                    os.path.join(self.pred_segment_dir, f"{stem}.txt")))
            )
            if has_pred:
                has_any_preds = True
        if has_any_preds:
            # Show only images that have prediction files
            for i, img_name in enumerate(self.images):
                stem = os.path.splitext(img_name)[0]
                has_pred = (
                    (self.pred_detect_dir and os.path.exists(
                        os.path.join(self.pred_detect_dir, f"{stem}.txt")))
                    or (self.pred_segment_dir and os.path.exists(
                        os.path.join(self.pred_segment_dir, f"{stem}.txt")))
                )
                if has_pred:
                    filtered.append(i)
        else:
            # No predictions anywhere — show images with annotations
            for i, img_name in enumerate(self.images):
                stem = os.path.splitext(img_name)[0]
                has_annot = (
                    (self.detect_dir and os.path.exists(
                        os.path.join(self.detect_dir, f"{stem}.txt")))
                    or (self.segment_dir and os.path.exists(
                        os.path.join(self.segment_dir, f"{stem}.txt")))
                )
                if has_annot:
                    filtered.append(i)
        self._review_filtered_images = filtered

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
        c = self.canvas
        c.bind("<Configure>", self._on_canvas_configure)
        c.bind("<ButtonPress-1>", self.on_button_press)
        c.bind("<B1-Motion>", self.on_move_press)
        c.bind("<ButtonRelease-1>", self.on_button_release)
        c.bind("<Double-Button-1>", self._on_double_click)
        c.bind("<Motion>", self._on_motion)

        if sys.platform == "darwin":
            # macOS: Button-2 is right-click, Button-3 is middle
            c.bind("<ButtonPress-2>", self.on_right_click)
            c.bind("<ButtonPress-3>", self.on_middle_press)
            c.bind("<B3-Motion>", self.on_middle_drag)
            c.bind("<ButtonRelease-3>", self.on_middle_release)
            # Control+click also triggers right-click on macOS
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

        r = self.root
        r.bind("<Right>", self._on_right_key)
        r.bind("<Left>", self._on_left_key)
        r.bind("<Up>", self._on_up_key)
        r.bind("<Down>", self._on_down_key)
        r.bind("<Escape>", self._on_escape)
        r.bind("<Control-z>", self.undo_last)
        r.bind("<Control-y>", self.redo_last)
        if sys.platform == "darwin":
            r.bind("<Command-z>", self.undo_last)
            r.bind("<Command-y>", self.redo_last)

        r.bind("h", lambda e: self._help_key())
        r.bind("m", lambda e: self._key_action(self._toggle_mode))
        r.bind("s", lambda e: self._annotate_key(self._toggle_snap))
        r.bind("v", lambda e: self._annotate_key(self._toggle_stream))

        r.bind("a", lambda e: self._review_key(self._review_accept))
        r.bind("r", lambda e: self._review_key(self._review_reject))
        r.bind("e", lambda e: self._review_key(self._review_edit))
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
            self._review_next_detection()
        else:
            self.next_image()

    def _on_left_key(self, event=None):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if self.tabview.get() == "Review":
            self._review_prev_detection()
        else:
            self.prev_image()

    def _on_up_key(self, event=None):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if self.tabview.get() == "Review":
            self._review_next_image()

    def _on_down_key(self, event=None):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
            return
        if self.tabview.get() == "Review":
            self._review_prev_image()

    def _help_key(self):
        """Toggle help in both tabs."""
        self._key_action(self.toggle_help)

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
        self.on_button_press(fake)

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
        self.display_image()
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
                self.save_annotations()
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
        self.load_image()

        # Reset and pre-load review for the new folder
        self._review_original_image = None
        self._review_index = 0
        self._review_detection_idx = 0
        self._review_needs_first_zoom = False
        if self._review_filtered_images:
            self._review_index = self._review_filtered_images[0]
            self._review_load_image()
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
                self.save_annotations()
                self._end_session()
                self._save_stats()
                print("[YoloLabeler] Done.")
            except Exception as e:
                print(f"Warning: Could not save on exit: {e}")
        if self._timer_after_id:
            self.root.after_cancel(self._timer_after_id)
        if self._resize_after_id:
            self.root.after_cancel(self._resize_after_id)
        if self._review_resize_after_id:
            self.root.after_cancel(self._review_resize_after_id)
        self.root.destroy()

    def _on_escape(self, event=None):
        if self.mode == "polygon":
            if self._stream_active:
                self._stream_active = False
                self._last_stream_pos = None
                self.display_image()
                return
            if self.current_polygon:
                self.current_polygon = []
                self.display_image()
                return
            if self._selected_polygon_idx is not None:
                self._selected_polygon_idx = None
                self._clear_drag_state()
                self.display_image()
                if self._review_return_pending:
                    self.root.after(50, self._review_confirm_dialog)

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
        if self.image_folder:
            return os.path.join(self.state_dir, "review_stats.json")
        return None

    def _load_review_state(self):
        path = self._review_state_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._review_state = json.load(f)
            except Exception:
                self._review_state = {}
        else:
            self._review_state = {}
        self._invalidate_reviewed_lookup()

    def _save_review_state(self):
        path = self._review_state_path()
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._review_state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save review state: {e}")

    def _mark_image_reviewed(self, img_name):
        """Mark an image as fully reviewed in review_stats.json."""
        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(img_name, {"img_status": "completed", "detections": []})
        img_data["img_status"] = "completed"
        self._save_review_state()

    def _is_image_reviewed(self, img_name):
        """Check if image has been reviewed."""
        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return False
        return img_data.get("img_status") == "completed"

    def _get_image_review_status(self, img_name):
        """Get review status string for an image: 'completed', 'started', or 'not_started'."""
        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return "not_started"
        return img_data.get("img_status", "not_started")

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
        self.save_annotations()
        self._save_stats()
        self._rebuild_filter()
        if self._filtered_indices:
            self.index = self._filtered_indices[0]
            self.load_image()
        else:
            # No images match filter — clear the canvas
            self.original_image = None
            self.canvas.delete("all")
            self._cached_scale = None
            self._cached_tk_image = None
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
                self.display_image()
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
                self.display_image()

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
                self._display_review_image()
            elif self.original_image is not None:
                self.display_image()
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
        self.save_annotations()
        self._save_stats()
        self.index = idx
        self.load_image()
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
    #  Coordinate conversions
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
        if not self.images or not self.image_folder:
            return
        if self.index >= len(self.images):
            self.index = 0
        if self.index < 0:
            self.index = len(self.images) - 1
        print(f"[YoloLabeler] Loading image {self.index + 1}/{len(self.images)}: {self.images[self.index]}")

        self.boxes = []
        self.polygons = []
        self._invalidate_poly_bboxes()
        self.current_polygon = []
        self._undo_stack = []
        self._redo_stack = []
        self._vertex_redo_stack = []
        self._dragging_vertex = None
        self._drag_orig_pos = None
        self._poly_preview_line = None
        self._snap_indicator_item = None
        self._stream_active = False
        self._last_stream_pos = None
        self._selected_polygon_idx = None
        self._hovered_polygon_idx = None
        self._annotate_pred_reference = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._cached_scale = None
        self._cached_tk_image = None

        self._image_start_time = None

        img_path = os.path.join(self.image_folder, self.images[self.index])

        # Try loading the image; skip corrupt files (bounded to avoid infinite loop)
        attempts = 0
        while attempts < len(self.images):
            img_path = os.path.join(self.image_folder,
                                    self.images[self.index])
            try:
                self.original_image = Image.open(img_path)
                self.original_image.load()
                self.original_image = auto_orient_image(self.original_image)
                break  # success
            except Exception as e:
                messagebox.showwarning(
                    "Image Error",
                    f"Could not load:\n{img_path}\n\n{e}")
                self.index += 1
                if self.index >= len(self.images):
                    self.index = 0
                attempts += 1
        else:
            # Every image failed to load
            messagebox.showerror(
                "No Valid Images",
                "No loadable images found in this folder.")
            return

        self.img_width, self.img_height = self.original_image.size
        self._image_dims[self.images[self.index]] = (
            self.img_width, self.img_height)

        self._initial_fit()
        self._load_existing_labels()
        # Record loaded count on first visit per session (baseline for delta)
        img_name = self.images[self.index]
        if img_name not in self._session_loaded_counts:
            self._session_loaded_counts[img_name] = (
                len(self.boxes) + len(self.polygons))
        if not getattr(self, '_defer_display', False):
            self.display_image()
        self.update_title()
        self._update_status()

    def _initial_fit(self):
        if self.img_width <= 0 or self.img_height <= 0:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10:
            cw = 1200
        if ch < 10:
            ch = 750
        sx = cw / self.img_width
        sy = ch / self.img_height
        fit_scale = min(sx, sy)
        self.zoom_index = self._nearest_zoom_index(fit_scale)
        self.scale = self.zoom_levels[self.zoom_index]
        self.offset_x = (cw - self.img_width * self.scale) / 2
        self.offset_y = (ch - self.img_height * self.scale) / 2

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
        if not self.detect_dir or not self.segment_dir:
            return
        stem = os.path.splitext(self.images[self.index])[0]

        detect_path = os.path.join(self.detect_dir, f"{stem}.txt")
        try:
            boxes, det_cids = parse_detect_labels(
                detect_path, self.img_width, self.img_height)
            self.boxes.extend(boxes)
            for cid in det_cids:
                if cid not in self.class_names:
                    self.class_names[cid] = f"class_{cid}"
                    self._refresh_class_dropdown()
                    self._save_classes_file()
        except Exception as e:
            print(f"Warning: Could not load detect labels for {stem}: {e}")

        segment_path = os.path.join(self.segment_dir, f"{stem}.txt")
        try:
            polygons, seg_cids = parse_segment_labels(
                segment_path, self.img_width, self.img_height)
            self.polygons.extend(polygons)
            self._invalidate_poly_bboxes()
            for cid in seg_cids:
                if cid not in self.class_names:
                    self.class_names[cid] = f"class_{cid}"
                    self._refresh_class_dropdown()
                    self._save_classes_file()
        except Exception as e:
            print(f"Warning: Could not load segment labels for {stem}: {e}")

    def _load_predictions(self, image_name, img_w, img_h):
        """Load model predictions for an image.

        Returns (pred_boxes, pred_polygons) where:
          pred_boxes:    [(x1, y1, x2, y2, class_id, conf), ...]
          pred_polygons: [(points, class_id, conf), ...]
        """
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

    # ──────────────────────────────────────────────────────────────────────────
    #  IoU matching engine — delegates to matching.py
    # ──────────────────────────────────────────────────────────────────────────

    _box_iou = staticmethod(box_iou)
    _polygon_iou_geom = staticmethod(polygon_iou)
    _box_to_points = staticmethod(box_to_points)

    def _compute_matches(self, gt_boxes, gt_polygons, pred_boxes, pred_polygons,
                         iou_threshold=0.5, conf_threshold=0.25):
        return compute_matches(gt_boxes, gt_polygons, pred_boxes,
                               pred_polygons, iou_threshold, conf_threshold)

    # ──────────────────────────────────────────────────────────────────────────
    #  Canvas resize debounce
    # ──────────────────────────────────────────────────────────────────────────
    def _on_canvas_configure(self, event=None):
        if self.original_image is None:
            return
        self._cached_scale = None          # invalidate cache (size changed)
        self._fast_resample = True          # use fast resampling while resizing
        self._initial_fit()                # re-fit image to new canvas size
        self._request_redraw()
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(200, self._finalize_resize)

    def _finalize_resize(self):
        self._resize_after_id = None
        self._fast_resample = False
        self._cached_scale = None          # force re-render at full quality
        self._initial_fit()                # final high-quality fit
        self.display_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Display (throttled)
    # ──────────────────────────────────────────────────────────────────────────
    def _request_redraw(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after_idle(self._do_redraw)

    def _do_redraw(self):
        self._redraw_pending = False
        self.display_image()

    def display_image(self):
        if self.original_image is None:
            return
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 800

        vis_x1, vis_y1 = self.canvas_to_image(0, 0)
        vis_x2, vis_y2 = self.canvas_to_image(cw, ch)

        crop_x1 = max(0, int(vis_x1))
        crop_y1 = max(0, int(vis_y1))
        crop_x2 = min(self.img_width, int(vis_x2) + 1)
        crop_y2 = min(self.img_height, int(vis_y2) + 1)

        cache_key = (self.scale, crop_x1, crop_y1, crop_x2, crop_y2)

        if self._cached_scale != cache_key:
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            if crop_w > 0 and crop_h > 0:
                cropped = self.original_image.crop(
                    (crop_x1, crop_y1, crop_x2, crop_y2))
                out_w = max(int(crop_w * self.scale), 1)
                out_h = max(int(crop_h * self.scale), 1)
                resized = cropped.resize(
                    (out_w, out_h),
                    Image.Resampling.BILINEAR if self._fast_resample
                    else Image.Resampling.LANCZOS)
                self._cached_tk_image = ImageTk.PhotoImage(
                    resized)
            else:
                self._cached_tk_image = None
            self._cached_scale = cache_key

        self.canvas.delete("all")
        self._snap_indicator_item = None

        if self._cached_tk_image is not None:
            place_x = self.offset_x + crop_x1 * self.scale
            place_y = self.offset_y + crop_y1 * self.scale
            self.canvas.create_image(
                place_x, place_y, anchor="nw", image=self._cached_tk_image)

        # Scale-dependent symbology
        s = self.scale
        line_w = max(1, min(2 + s * 0.5, 6))
        poly_w = max(1, min(2.5 + s * 0.5, 7))
        vert_r = max(3, min(VERTEX_HANDLE_RADIUS * (1.6 - s * 0.2), 12))
        sel_vert_r = max(vert_r + 2, min(VERTEX_HANDLE_RADIUS * (2.2 - s * 0.2), 16))
        label_size = max(7, min(int(9 * (0.6 + s * 0.4)), 18))
        dash_a = max(2, int(4 * (0.5 + s * 0.5)))
        dash_b = max(2, int(4 * (0.5 + s * 0.5)))

        def _halo(x, y, text, fill, **kw):
            """Draw text with dark outline for readability on any background."""
            for dx, dy in [(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),
                           (-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),
                           (0,-2),(0,-1),(0,1),(0,2),
                           (1,-2),(1,-1),(1,0),(1,1),(1,2),
                           (2,-2),(2,-1),(2,0),(2,1),(2,2)]:
                self.canvas.create_text(x + dx, y + dy, text=text,
                                        fill="black", **kw)
            self.canvas.create_text(x, y, text=text, fill=fill, **kw)

        if self.mode == "box" and self._annotation_visible:
            for box in self.boxes:
                x1, y1, x2, y2, class_id = box
                if class_id != self.active_class:
                    continue
                # Viewport culling — skip boxes entirely outside visible region
                if x2 < vis_x1 or x1 > vis_x2 or y2 < vis_y1 or y1 > vis_y2:
                    continue
                cx1, cy1 = self.image_to_canvas(x1, y1)
                cx2, cy2 = self.image_to_canvas(x2, y2)
                color = self._get_class_color(class_id)
                self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2, outline=color, width=line_w)
                class_name = self.class_names.get(class_id, str(class_id))
                _halo(cx1 + 2, cy1 - 2, anchor="sw",
                      text=f"{class_id}: {class_name}",
                      fill=color, font=(self.font_family, label_size, "bold"))

        if self.mode == "polygon" and self._annotation_visible:
            for poly_idx, (points, class_id) in enumerate(self.polygons):
                is_selected = (poly_idx == self._selected_polygon_idx)
                if class_id != self.active_class and not is_selected:
                    continue
                # Viewport culling — skip polygons entirely outside visible region
                if points and not is_selected:
                    pxs = [p[0] for p in points]
                    pys = [p[1] for p in points]
                    if max(pxs) < vis_x1 or min(pxs) > vis_x2 or max(pys) < vis_y1 or min(pys) > vis_y2:
                        continue
                color = self._get_class_color(class_id)
                draw_color = "#00BFFF" if is_selected else color
                canvas_pts = []
                for px, py in points:
                    cx, cy = self.image_to_canvas(px, py)
                    canvas_pts.extend([cx, cy])
                if len(canvas_pts) >= 6:
                    self.canvas.create_polygon(
                        *canvas_pts, outline=draw_color, fill="",
                        width=poly_w)
                # Show vertices: selected polygon always, hovered, or dragged
                show_verts = (
                    is_selected
                    or poly_idx == self._hovered_polygon_idx
                    or (self._dragging_vertex is not None
                        and self._dragging_vertex[0] == poly_idx)
                )
                # During snap-drag, also show vertices on nearby polygons
                if (not show_verts
                        and self._dragging_vertex is not None
                        and self.snap_enabled
                        and points):
                    dpi = self._dragging_vertex[0]
                    if dpi < len(self.polygons):
                        dvi = self._dragging_vertex[1]
                        dpts = self.polygons[dpi][0]
                        if dvi < len(dpts):
                            dcx, dcy = self.image_to_canvas(*dpts[dvi])
                            for px, py in points:
                                pcx, pcy = self.image_to_canvas(px, py)
                                if math.hypot(dcx - pcx, dcy - pcy) < SNAP_RADIUS * 3:
                                    show_verts = True
                                    break
                if show_verts:
                    r = sel_vert_r if is_selected else vert_r
                    for px, py in points:
                        cx, cy = self.image_to_canvas(px, py)
                        self.canvas.create_oval(
                            cx - r, cy - r,
                            cx + r, cy + r,
                            fill=draw_color, outline="white", width=1)
                if points:
                    lx, ly = self.image_to_canvas(*points[0])
                    class_name = self.class_names.get(class_id, str(class_id))
                    _halo(lx + 2, ly - 2, anchor="sw",
                          text=f"{class_id}: {class_name}",
                          fill=draw_color, font=(self.font_family, label_size, "bold"))

        if self.current_polygon:
            color = self._get_class_color(self.active_class)
            for i, (px, py) in enumerate(self.current_polygon):
                cx, cy = self.image_to_canvas(px, py)
                self.canvas.create_oval(
                    cx - vert_r, cy - vert_r,
                    cx + vert_r, cy + vert_r,
                    fill=color, outline="white", width=1)
                if i > 0:
                    prev_cx, prev_cy = self.image_to_canvas(
                        *self.current_polygon[i - 1])
                    self.canvas.create_line(
                        prev_cx, prev_cy, cx, cy,
                        fill=color, width=line_w, dash=(dash_a, dash_b))
            last_cx, last_cy = self.image_to_canvas(
                *self.current_polygon[-1])
            self._poly_preview_line = self.canvas.create_line(
                last_cx, last_cy,
                self._mouse_canvas_x, self._mouse_canvas_y,
                fill=color, width=max(1, line_w * 0.5),
                dash=(dash_a // 2 or 1, dash_b))

        # ── Prediction reference overlay (from Review tab) ──
        if self._annotate_pred_reference:
            ref = self._annotate_pred_reference
            PRED_REF_COLOR = "#00BFFF"
            ref_dash = (6, 4)
            ref_lw = max(1, min(2 + s * 0.5, 5))
            ref_label_size = max(8, min(int(10 * (0.6 + s * 0.4)), 16))
            cid = ref.get('class_id', 0)
            conf = ref.get('conf', 0)
            name = self.class_names.get(cid, str(cid))

            if ref['type'] == 'box':
                x1, y1, x2, y2 = ref['coords']
                cx1, cy1 = self.image_to_canvas(x1, y1)
                cx2, cy2 = self.image_to_canvas(x2, y2)
                self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline=PRED_REF_COLOR, width=ref_lw,
                    dash=ref_dash)
                _halo(cx1 + 2, cy1 - 2, anchor="sw",
                      text=f"Pred {cid}: {name} ({conf:.2f})",
                      fill=PRED_REF_COLOR,
                      font=(self.font_family, ref_label_size, "bold"))
            elif ref['type'] == 'polygon':
                pts = ref['coords']
                canvas_pts = []
                for px_pt, py_pt in pts:
                    cx_p, cy_p = self.image_to_canvas(px_pt, py_pt)
                    canvas_pts.extend([cx_p, cy_p])
                if len(canvas_pts) >= 6:
                    self.canvas.create_polygon(
                        *canvas_pts, outline=PRED_REF_COLOR,
                        fill="", width=ref_lw, dash=ref_dash)
                if pts:
                    lx, ly = self.image_to_canvas(*pts[0])
                    _halo(lx + 2, ly - 2, anchor="sw",
                          text=f"Pred {cid}: {name} ({conf:.2f})",
                          fill=PRED_REF_COLOR,
                          font=(self.font_family, ref_label_size, "bold"))

        self.draw_help_overlay()

    # ──────────────────────────────────────────────────────────────────────────
    #  Help overlay — grouped by Keyboard / Mouse
    # ──────────────────────────────────────────────────────────────────────────
    def draw_help_overlay(self):
        if not self.show_help:
            return

        if self.mode == "box":
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

        # Measure text precisely using tkFont
        fnt = tkFont.Font(family=font_family, size=font_size)
        line_height = fnt.metrics("linespace") + 2
        max_text_w = max(fnt.measure(ln) for ln in help_lines) if help_lines else 100

        block_w = max_text_w + pad * 3
        block_h = len(help_lines) * line_height + pad * 2
        x0, y0 = 10, 10

        self.canvas.create_rectangle(
            x0, y0, x0 + block_w, y0 + block_h,
            fill="#1A1A1A", outline="#444444", width=1, stipple="")

        for i, line in enumerate(help_lines):
            self.canvas.create_text(
                x0 + pad, y0 + pad + i * line_height,
                anchor="nw", text=line,
                fill=FG_COLOR, font=(font_family, font_size))

    def toggle_help(self, event=None):
        if self.tabview.get() == "Review":
            self._review_show_help = not self._review_show_help
            self._display_review_image()
        else:
            self.show_help = not self.show_help
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
        self._update_status()

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
        self._mouse_canvas_x = event.x
        self._mouse_canvas_y = event.y

        # Streaming: auto-place vertices as mouse moves
        if (self.mode == "polygon" and self._stream_mode
                and self._stream_active and self.current_polygon):
            ix, iy = self.canvas_to_image(event.x, event.y)
            ix = max(0, min(self.img_width, ix))
            iy = max(0, min(self.img_height, iy))
            # Edge snap: project cursor onto nearest polygon edge
            snapped_ix, snapped_iy = self._snap_to_edge(ix, iy)
            if (snapped_ix, snapped_iy) != (ix, iy):
                # Near a polygon edge — place snapped vertex if not duplicate
                if (self.current_polygon[-1] != (snapped_ix, snapped_iy)):
                    self.current_polygon.append((snapped_ix, snapped_iy))
                    self._last_stream_pos = (snapped_ix, snapped_iy)
                    self.display_image()
            elif self._last_stream_pos:
                dist = math.hypot(
                    ix - self._last_stream_pos[0],
                    iy - self._last_stream_pos[1])
                if dist >= STREAM_MIN_DISTANCE:
                    self.current_polygon.append((ix, iy))
                    self._last_stream_pos = (ix, iy)
                    self.display_image()
            # Fall through to snap indicator + hover (don't return)

        # Throttle expensive hover/snap checks (~60fps cap)
        now = time.monotonic()
        _motion_throttled = (now - self._motion_last_time) < 0.016

        # Update snap indicator
        if not _motion_throttled and self.mode == "polygon" and self.snap_enabled:
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
        elif not (self.mode == "polygon" and self.snap_enabled):
            if self._snap_indicator_item:
                try:
                    self.canvas.delete(self._snap_indicator_item)
                except tk.TclError:
                    pass
                self._snap_indicator_item = None

        # Polygon hover detection (for showing vertices on adjacent polygons)
        if not _motion_throttled and self.mode == "polygon":
            self._motion_last_time = now
            ix, iy = self.canvas_to_image(event.x, event.y)
            new_hover = None
            hover_thr = 25  # canvas-pixel proximity to reveal vertices
            if not self.current_polygon:
                # Not drawing — full hit-test (vertex, edge, proximity, point-in-polygon)
                vhit = self._find_nearest_vertex(event.x, event.y, threshold=hover_thr)
                if vhit:
                    new_hover = vhit[0]
                else:
                    ehit = self._find_nearest_edge_selected(event.x, event.y, threshold=hover_thr)
                    if ehit is not None:
                        new_hover = ehit
                    else:
                        for pi, (points, _) in enumerate(self.polygons):
                            if self._point_in_polygon(ix, iy, points):
                                new_hover = pi
                                break
            else:
                # Drawing — show vertices on nearest polygon for snap visibility
                vhit = self._find_nearest_vertex(event.x, event.y, threshold=hover_thr + 5)
                if vhit:
                    new_hover = vhit[0]
                else:
                    ehit = self._find_nearest_edge_selected(event.x, event.y, threshold=hover_thr)
                    if ehit is not None:
                        new_hover = ehit
                    else:
                        for pi, (points, _) in enumerate(self.polygons):
                            if self._point_in_polygon(ix, iy, points):
                                new_hover = pi
                                break
            if new_hover != self._hovered_polygon_idx:
                self._hovered_polygon_idx = new_hover
                self._request_redraw()
        elif self.mode != "polygon":
            if self._hovered_polygon_idx is not None:
                self._hovered_polygon_idx = None

        # Polygon preview line
        if self.mode == "polygon" and self.current_polygon:
            if self._poly_preview_line is not None:
                try:
                    last_cx, last_cy = self.image_to_canvas(
                        *self.current_polygon[-1])
                    self.canvas.coords(
                        self._poly_preview_line,
                        last_cx, last_cy, event.x, event.y)
                except tk.TclError:
                    pass

    # ──────────────────────────────────────────────────────────────────────────
    #  Vertex snapping
    # ──────────────────────────────────────────────────────────────────────────
    def _maybe_snap(self, ix, iy, exclude=None):
        if not self.snap_enabled:
            return (ix, iy)
        self._ensure_poly_bboxes()
        cx, cy = self.image_to_canvas(ix, iy)
        img_thr = SNAP_RADIUS / self.scale if self.scale > 0 else 1e9
        best_dist = SNAP_RADIUS
        best_pt = None
        for pidx, (points, _) in enumerate(self.polygons):
            if pidx < len(self._poly_bboxes):
                bx1, by1, bx2, by2 = self._poly_bboxes[pidx]
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
        """Snap to the nearest polygon edge (line segment) in canvas space.
        Returns the projected image-coordinate point on the edge,
        or the original point if nothing is within SNAP_RADIUS."""
        if not self.snap_enabled:
            return (ix, iy)
        self._ensure_poly_bboxes()
        cx, cy = self.image_to_canvas(ix, iy)
        img_thr = SNAP_RADIUS / self.scale if self.scale > 0 else 1e9
        best_dist = SNAP_RADIUS
        best_pt = None
        for pidx, (points, _) in enumerate(self.polygons):
            if pidx < len(self._poly_bboxes):
                bx1, by1, bx2, by2 = self._poly_bboxes[pidx]
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
                    # Convert projected canvas point back to image coords
                    pix, piy = self.canvas_to_image(proj_cx, proj_cy)
                    pix = max(0, min(self.img_width, pix))
                    piy = max(0, min(self.img_height, piy))
                    best_pt = (pix, piy)
        if best_pt:
            return best_pt
        return (ix, iy)

    # ──────────────────────────────────────────────────────────────────────────
    #  Mouse event dispatch
    # ──────────────────────────────────────────────────────────────────────────
    def on_button_press(self, event):
        if self.mode == "box":
            self._box_press(event)
        else:
            self._poly_press(event)

    def on_move_press(self, event):
        if self.mode == "box":
            self._box_drag(event)
        else:
            self._poly_drag(event)

    def on_button_release(self, event):
        if self.mode == "box":
            self._box_release(event)
        else:
            self._poly_release(event)

    def _on_double_click(self, event):
        if self.mode != "polygon":
            return
        # If drawing, close the polygon
        if self.current_polygon:
            self._stream_active = False
            self._last_stream_pos = None
            if len(self.current_polygon) >= 3:
                self._close_polygon()
            return

    def _clear_drag_state(self):
        self._dragging_vertex = None
        self._drag_orig_pos = None
        self._hovered_polygon_idx = None
        self.canvas.config(cursor="cross")

    def on_right_click(self, event):
        if self.mode == "polygon" and self.current_polygon:
            self.current_polygon = []
            self._stream_active = False
            self._last_stream_pos = None
            self.display_image()
            return

        click_ix, click_iy = self.canvas_to_image(event.x, event.y)

        if self.mode == "polygon" and self._selected_polygon_idx is not None:
            # Right-click scoped to selected polygon
            pi = self._selected_polygon_idx
            if pi < len(self.polygons):
                vertex_hit = self._find_nearest_vertex(
                    event.x, event.y, threshold=10)
                if vertex_hit and vertex_hit[0] == pi:
                    vi = vertex_hit[1]
                    points, cls = self.polygons[pi]
                    self._push_undo()
                    if len(points) <= 3:
                        self.polygons.pop(pi)
                        self._selected_polygon_idx = None
                    else:
                        new_pts = list(points)
                        new_pts.pop(vi)
                        self.polygons[pi] = (new_pts, cls)
                    self._invalidate_poly_bboxes()
                    self._clear_drag_state()
                    self._mark_image_annotated()
                    self.display_image()
                    self.update_title()
                    return
                # Right-click inside selected polygon = delete it
                if self._point_in_polygon(click_ix, click_iy,
                                          self.polygons[pi][0]):
                    self._push_undo()
                    self.polygons.pop(pi)
                    self._invalidate_poly_bboxes()
                    self._selected_polygon_idx = None
                    self._clear_drag_state()
                    self._mark_image_annotated()
                    self.display_image()
                    self.update_title()
                    return
            # Click outside selected polygon = deselect
            self._selected_polygon_idx = None
            self.display_image()
            if self._review_return_pending:
                self.root.after(50, self._review_confirm_dialog)
            return

        for i, (x1, y1, x2, y2, _) in enumerate(self.boxes):
            if x1 <= click_ix <= x2 and y1 <= click_iy <= y2:
                self._push_undo()
                self.boxes.pop(i)
                self._mark_image_annotated()
                self.display_image()
                self.update_title()
                return

        for i, (points, _) in enumerate(self.polygons):
            if self._point_in_polygon(click_ix, click_iy, points):
                self._push_undo()
                self._clear_drag_state()
                self.polygons.pop(i)
                self._invalidate_poly_bboxes()
                self._selected_polygon_idx = None
                self._mark_image_annotated()
                self.display_image()
                self.update_title()
                return

    # ──────────────────────────────────────────────────────────────────────────
    #  Box mode
    # ──────────────────────────────────────────────────────────────────────────
    def _box_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        color = self._get_class_color(self.active_class)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=color, width=2)

    def _box_drag(self, event):
        if (self.rect and self.start_x is not None
                and self.start_y is not None):
            self.canvas.coords(
                self.rect, self.start_x, self.start_y, event.x, event.y)

    def _box_release(self, event):
        if self.start_x is None or self.start_y is None:
            return
        ix1, iy1 = self.canvas_to_image(self.start_x, self.start_y)
        ix2, iy2 = self.canvas_to_image(event.x, event.y)

        x1 = max(0, min(ix1, ix2))
        y1 = max(0, min(iy1, iy2))
        x2 = min(self.img_width, max(ix1, ix2))
        y2 = min(self.img_height, max(iy1, iy2))

        if (x2 - x1) < 3 or (y2 - y1) < 3:
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = None
            return

        self._push_undo()
        self.boxes.append((x1, y1, x2, y2, self.active_class))
        self._mark_image_annotated()
        self._record_annotation_added()
        self.rect = None
        self.display_image()
        self.update_title()

        # Auto-prompt when editing from Review
        if self._review_return_pending:
            self.root.after(50, self._review_confirm_dialog)

    # ──────────────────────────────────────────────────────────────────────────
    #  Polygon mode
    # ──────────────────────────────────────────────────────────────────────────
    def _poly_press(self, event):
        ix, iy = self.canvas_to_image(event.x, event.y)

        # ── Currently drawing a polygon ──
        if self.current_polygon:
            if self._stream_mode:
                if self._stream_active:
                    self._stream_active = False
                    self._last_stream_pos = None
                    self.display_image()
                else:
                    snapped = self._maybe_snap(ix, iy)
                    ix, iy = snapped
                    ix = max(0, min(self.img_width, ix))
                    iy = max(0, min(self.img_height, iy))
                    self.current_polygon.append((ix, iy))
                    self._stream_active = True
                    self._last_stream_pos = (ix, iy)
                    self._vertex_redo_stack.clear()
                    self.display_image()
            else:
                snapped = self._maybe_snap(ix, iy)
                ix, iy = snapped
                ix = max(0, min(self.img_width, ix))
                iy = max(0, min(self.img_height, iy))
                self.current_polygon.append((ix, iy))
                self._vertex_redo_stack.clear()
                self.display_image()
            return

        # ── A polygon is selected for editing ──
        if self._selected_polygon_idx is not None:
            pi = self._selected_polygon_idx
            if pi < len(self.polygons):
                # Vertex drag — check only this polygon's vertices so
                # a coincident vertex on another polygon can't steal the hit
                best_vi, best_vd = None, 8
                for vi, (px, py) in enumerate(self.polygons[pi][0]):
                    vcx, vcy = self.image_to_canvas(px, py)
                    d = math.hypot(event.x - vcx, event.y - vcy)
                    if d < best_vd:
                        best_vd = d
                        best_vi = vi
                if best_vi is not None:
                    self._push_undo()
                    self._dragging_vertex = (pi, best_vi)
                    self._drag_orig_pos = self.polygons[pi][0][best_vi]
                    self.canvas.config(cursor="fleur")
                    return
                # Edge insert — check only this polygon's edges
                best_ei, best_ed, best_ept = None, 6, None
                pts_sel = self.polygons[pi][0]
                n_sel = len(pts_sel)
                for ei in range(n_sel):
                    ax, ay = self.image_to_canvas(*pts_sel[ei])
                    bx, by = self.image_to_canvas(*pts_sel[(ei + 1) % n_sel])
                    d = _point_to_segment_dist(event.x, event.y, ax, ay, bx, by)
                    if d < best_ed:
                        best_ed = d
                        # Project click onto edge
                        edx, edy = bx - ax, by - ay
                        len_sq = edx * edx + edy * edy
                        if len_sq == 0:
                            proj_cx, proj_cy = ax, ay
                        else:
                            t = max(0.0, min(1.0, ((event.x - ax) * edx + (event.y - ay) * edy) / len_sq))
                            proj_cx = ax + t * edx
                            proj_cy = ay + t * edy
                        pix, piy = self.canvas_to_image(proj_cx, proj_cy)
                        pix = max(0, min(self.img_width, pix))
                        piy = max(0, min(self.img_height, piy))
                        best_ei = ei
                        best_ept = (pix, piy)
                if best_ei is not None:
                    self._push_undo()
                    points, cls = self.polygons[pi]
                    new_points = list(points)
                    new_points.insert(best_ei + 1, best_ept)
                    self.polygons[pi] = (new_points, cls)
                    self._invalidate_poly_bboxes()
                    self._dragging_vertex = (pi, best_ei + 1)
                    self._drag_orig_pos = best_ept
                    self.canvas.config(cursor="fleur")
                    self.display_image()
                    return
            # Click not on selected polygon's vertex/edge — deselect
            # and fall through to check if clicking on another polygon
            just_deselected = True
            self._selected_polygon_idx = None
            if self._review_return_pending:
                self.display_image()
                self.root.after(50, self._review_confirm_dialog)
                return
        else:
            just_deselected = False

        # ── Not drawing, nothing selected — check if clicking on polygon ──
        # When snap is enabled, skip vertex-proximity selection so the user
        # can start a new polygon snapped to an existing vertex.  They can
        # still select by clicking *inside* the polygon body.
        if not self.snap_enabled:
            vhit = self._find_nearest_vertex(event.x, event.y, threshold=15)
            if vhit:
                self._selected_polygon_idx = vhit[0]
                self.display_image()
                return
        for pi, (points, _) in enumerate(self.polygons):
            if self._point_in_polygon(ix, iy, points):
                self._selected_polygon_idx = pi
                self.display_image()
                return

        # If we just deselected, only redraw — don't start a new polygon
        if just_deselected:
            self.display_image()
            return

        # ── Start new polygon (empty space) ──
        ix, iy = self._maybe_snap(ix, iy)
        ix = max(0, min(self.img_width, ix))
        iy = max(0, min(self.img_height, iy))
        self.current_polygon = [(ix, iy)]

        if self._stream_mode:
            self._stream_active = True
            self._last_stream_pos = (ix, iy)

        self.display_image()

    def _poly_drag(self, event):
        if self._dragging_vertex is not None:
            pi, vi = self._dragging_vertex
            if pi >= len(self.polygons):
                self._clear_drag_state()
                return
            # Only allow drag on the selected polygon
            if pi != self._selected_polygon_idx:
                self._clear_drag_state()
                return
            raw_ix, raw_iy = self.canvas_to_image(event.x, event.y)
            ix, iy = self._maybe_snap(raw_ix, raw_iy, exclude=(pi, vi))
            did_snap = (ix, iy) != (raw_ix, raw_iy)
            ix = max(0, min(self.img_width, ix))
            iy = max(0, min(self.img_height, iy))
            points, cls = self.polygons[pi]
            new_points = list(points)
            new_points[vi] = (ix, iy)
            self.polygons[pi] = (new_points, cls)
            self._invalidate_poly_bboxes()
            self.display_image()
            # Show snap indicator during drag (only on target vertex)
            if self.snap_enabled:
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
        if self._dragging_vertex is not None:
            self._mark_image_annotated()
            self._dragging_vertex = None
            self._drag_orig_pos = None
            self.canvas.config(cursor="cross")

    def _close_polygon(self):
        if len(self.current_polygon) < 3:
            self.current_polygon = []
            self.display_image()
            return
        clamped = []
        for x, y in self.current_polygon:
            cx = max(0, min(self.img_width, x))
            cy = max(0, min(self.img_height, y))
            clamped.append((cx, cy))
        self._push_undo()
        self.polygons.append((clamped, self.active_class))
        self._invalidate_poly_bboxes()
        self._mark_image_annotated()
        self._record_annotation_added()
        self.current_polygon = []
        self._poly_preview_line = None
        self.display_image()
        self.update_title()

        # Auto-prompt when editing from Review
        if self._review_return_pending:
            self.root.after(50, self._review_confirm_dialog)

    # ──────────────────────────────────────────────────────────────────────────
    #  Polygon spatial index — precomputed bounding boxes
    # ──────────────────────────────────────────────────────────────────────────
    def _invalidate_poly_bboxes(self):
        self._poly_bboxes_dirty = True

    def _ensure_poly_bboxes(self):
        if not self._poly_bboxes_dirty:
            return
        self._poly_bboxes = []
        for points, _ in self.polygons:
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                self._poly_bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                self._poly_bboxes.append((0, 0, 0, 0))
        self._poly_bboxes_dirty = False

    # ──────────────────────────────────────────────────────────────────────────
    #  Polygon geometry helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _find_nearest_vertex(self, cx, cy, threshold=8):
        self._ensure_poly_bboxes()
        qix, qiy = self.canvas_to_image(cx, cy)
        img_thr = threshold / self.scale if self.scale > 0 else 1e9
        best = None
        best_dist = threshold
        for pi, (points, _) in enumerate(self.polygons):
            if pi < len(self._poly_bboxes):
                bx1, by1, bx2, by2 = self._poly_bboxes[pi]
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
        self._ensure_poly_bboxes()
        qix, qiy = self.canvas_to_image(cx, cy)
        img_thr = threshold / self.scale if self.scale > 0 else 1e9
        best = None
        best_dist = threshold
        for pi, (points, _) in enumerate(self.polygons):
            if pi < len(self._poly_bboxes):
                bx1, by1, bx2, by2 = self._poly_bboxes[pi]
                if (qix + img_thr < bx1 or qix - img_thr > bx2
                        or qiy + img_thr < by1 or qiy - img_thr > by2):
                    continue
            n = len(points)
            for ei in range(n):
                ax, ay = self.image_to_canvas(*points[ei])
                bx, by = self.image_to_canvas(*points[(ei + 1) % n])
                dist = _point_to_segment_dist(cx, cy, ax, ay, bx, by)
                if dist < best_dist:
                    best_dist = dist
                    ix, iy = self.canvas_to_image(cx, cy)
                    ix = max(0, min(self.img_width, ix))
                    iy = max(0, min(self.img_height, iy))
                    best = (pi, ei, (ix, iy))
        return best

    def _find_nearest_edge_selected(self, cx, cy, threshold=6):
        """Like _find_nearest_edge but only checks the selected polygon.

        Returns the polygon index (int) if an edge is within threshold,
        or None.  Used for hover detection so coincident edges on other
        polygons don't steal the hover highlight.
        """
        pi = self._selected_polygon_idx
        if pi is None or pi >= len(self.polygons):
            return None
        points = self.polygons[pi][0]
        n = len(points)
        for ei in range(n):
            ax, ay = self.image_to_canvas(*points[ei])
            bx, by = self.image_to_canvas(*points[(ei + 1) % n])
            dist = _point_to_segment_dist(cx, cy, ax, ay, bx, by)
            if dist < threshold:
                return pi
        return None

    _point_in_polygon = staticmethod(point_in_polygon)

    # ══════════════════════════════════════════════════════════════════════════
    #  Review Tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_review_tab(self):
        """Build the Review tab UI inside self._review_frame.

        All controls (filters, sliders, action buttons, detection nav) now
        live in the main toolbar and status bar.  This method only creates
        the review canvas.
        """
        # ── Review canvas ──
        self._review_canvas = tk.Canvas(self._review_frame, bg=CANVAS_BG,
                                        highlightthickness=0)
        self._review_canvas.pack(fill="both", expand=True)
        self._setup_review_bindings()

    def _setup_review_bindings(self):
        """Canvas bindings for the review tab (zoom and pan only)."""
        c = self._review_canvas
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

    def _refresh_review_class_filter(self):
        """Update class filter dropdown values from current class_names."""
        vals = ["All"] + [f"{cid}: {name}"
                          for cid, name in sorted(self.class_names.items())]
        self._review_class_dd.configure(values=vals)

    # ── Review coordinate conversion ─────────────────────────────────────────

    def _review_image_to_canvas(self, ix, iy):
        return (ix * self._review_scale + self._review_offset_x,
                iy * self._review_scale + self._review_offset_y)

    def _review_canvas_to_image(self, cx, cy):
        s = self._review_scale if self._review_scale else 1.0
        return ((cx - self._review_offset_x) / s,
                (cy - self._review_offset_y) / s)

    # ── Review pan / zoom ────────────────────────────────────────────────────

    def _review_pan_start(self, event):
        self._review_pan_start_x = event.x
        self._review_pan_start_y = event.y

    def _review_pan_drag(self, event):
        if self._review_pan_start_x is None:
            return
        dx = event.x - self._review_pan_start_x
        dy = event.y - self._review_pan_start_y
        self._review_offset_x += dx
        self._review_offset_y += dy
        self._review_pan_start_x = event.x
        self._review_pan_start_y = event.y
        self._display_review_image()

    def _review_zoom(self, event):
        cx, cy = self._review_canvas_to_image(event.x, event.y)
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        new_scale = max(0.05, min(10.0, self._review_scale * factor))
        self._review_offset_x = event.x - cx * new_scale
        self._review_offset_y = event.y - cy * new_scale
        self._review_scale = new_scale
        self._review_cached_scale = None
        self._display_review_image()
        self._update_status()

    def _review_scroll(self, event):
        if sys.platform == "darwin":
            delta = -event.delta * 2
        else:
            delta = -event.delta // 3
        self._review_offset_y -= delta
        self._display_review_image()

    def _review_hscroll(self, event):
        if sys.platform == "darwin":
            delta = -event.delta * 2
        else:
            delta = -event.delta // 3
        self._review_offset_x -= delta
        self._display_review_image()

    def _review_scroll_linux(self, event, direction):
        self._review_offset_y += direction * 40
        self._display_review_image()

    def _review_hscroll_linux(self, event, direction):
        self._review_offset_x += direction * 40
        self._display_review_image()

    def _review_zoom_linux(self, event, direction):
        cx, cy = self._review_canvas_to_image(event.x, event.y)
        factor = 1.15 if direction > 0 else 1 / 1.15
        new_scale = max(0.05, min(10.0, self._review_scale * factor))
        self._review_offset_x = event.x - cx * new_scale
        self._review_offset_y = event.y - cy * new_scale
        self._review_scale = new_scale
        self._review_cached_scale = None
        self._display_review_image()
        self._update_status()

    def _on_review_canvas_configure(self, event=None):
        """Redraw review canvas on window resize (debounced)."""
        if self._review_original_image is None:
            return
        self._review_cached_scale = None
        if self._review_resize_after_id is not None:
            self.root.after_cancel(self._review_resize_after_id)
        self._review_resize_after_id = self.root.after(
            200, self._finalize_review_resize)

    def _finalize_review_resize(self):
        self._review_resize_after_id = None
        self._review_cached_scale = None
        self._display_review_image()

    # ── Review image loading ─────────────────────────────────────────────────

    def _review_load_image(self):
        """Load image, GT, predictions, and run matching for review."""
        if not self.images:
            return
        # Record review time for previous image before loading new one
        self._record_review_time()
        self._review_image_start_time = time.time()
        idx = self._review_index
        if idx < 0 or idx >= len(self.images):
            return
        img_name = self.images[idx]
        img_path = os.path.join(self.image_folder, img_name)

        try:
            pil_img = Image.open(img_path)
            pil_img = auto_orient_image(pil_img)
            pil_img = pil_img.convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image for review: {e}")
            return

        self._review_original_image = pil_img
        self._review_img_w = pil_img.width
        self._review_img_h = pil_img.height

        # Load GT from disk
        self._review_gt_boxes = []
        self._review_gt_polygons = []
        stem = os.path.splitext(img_name)[0]

        detect_path = os.path.join(self.detect_dir, f"{stem}.txt")
        try:
            boxes, _ = parse_detect_labels(
                detect_path, self._review_img_w, self._review_img_h)
            self._review_gt_boxes.extend(boxes)
        except Exception:
            pass

        segment_path = os.path.join(self.segment_dir, f"{stem}.txt")
        try:
            polys, _ = parse_segment_labels(
                segment_path, self._review_img_w, self._review_img_h)
            self._review_gt_polygons.extend(polys)
        except Exception:
            pass

        # Load predictions
        self._review_pred_boxes, self._review_pred_polygons = \
            self._load_predictions(
                img_name, self._review_img_w, self._review_img_h)

        # Run matching
        self._review_matches = self._compute_matches(
            self._review_gt_boxes, self._review_gt_polygons,
            self._review_pred_boxes, self._review_pred_polygons,
            iou_threshold=REVIEW_IOU_THRESHOLD,
            conf_threshold=REVIEW_CONF_THRESHOLD)

        self._rebuild_review_detections()
        self._refresh_review_class_filter()

        # Force geometry update so canvas dimensions are accurate
        self._review_canvas.update_idletasks()

        # Fit image to canvas (use scheduled dimensions if canvas not ready)
        cw = self._review_canvas.winfo_width()
        ch = self._review_canvas.winfo_height()
        if cw < 10 or ch < 10:
            # Canvas not realized yet — schedule a deferred load
            self.root.after(50, self._review_deferred_zoom)
            return
        sx = cw / max(self._review_img_w, 1)
        sy = ch / max(self._review_img_h, 1)
        self._review_scale = min(sx, sy)
        self._review_offset_x = (cw - self._review_img_w * self._review_scale) / 2
        self._review_offset_y = (ch - self._review_img_h * self._review_scale) / 2
        self._review_cached_scale = None

        # Zoom to first unreviewed detection (or first if all reviewed)
        self._review_zoom_to_first_unreviewed()

        self._update_review_labels()

    def _review_deferred_zoom(self):
        """Deferred zoom after canvas geometry is available."""
        cw = self._review_canvas.winfo_width()
        ch = self._review_canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.root.after(50, self._review_deferred_zoom)
            return
        sx = cw / max(self._review_img_w, 1)
        sy = ch / max(self._review_img_h, 1)
        self._review_scale = min(sx, sy)
        self._review_offset_x = (cw - self._review_img_w * self._review_scale) / 2
        self._review_offset_y = (ch - self._review_img_h * self._review_scale) / 2
        self._review_cached_scale = None
        self._review_zoom_to_first_unreviewed()
        self._update_review_labels()

    def _review_zoom_to_first_unreviewed(self):
        """Zoom to first unreviewed detection, or first if all reviewed."""
        if self._review_detections:
            img_name = self.images[self._review_index] if self.images else ""
            first_unreviewed = 0
            for i, det in enumerate(self._review_detections):
                if not self._find_reviewed_entry(det, img_name):
                    first_unreviewed = i
                    break
            self._review_detection_idx = first_unreviewed
            self._zoom_to_detection()
        else:
            self._review_detection_idx = 0
            self._display_review_image()

    # ── Review detection management ──────────────────────────────────────────

    def _current_review_det(self):
        """Return the currently focused detection dict, or None."""
        if (self._review_detections
                and 0 <= self._review_detection_idx
                < len(self._review_detections)):
            return self._review_detections[self._review_detection_idx]
        return None

    def _refind_detection(self, prev_det):
        """Find the index of prev_det in the current detection list.

        Matches by gt/pred type and index.  Returns 0 if not found.
        """
        if prev_det is None or not self._review_detections:
            return 0
        gt_t = prev_det.get('gt_type')
        gt_i = prev_det.get('gt_idx')
        pt_t = prev_det.get('pred_type')
        pt_i = prev_det.get('pred_idx')
        for i, d in enumerate(self._review_detections):
            if (d.get('gt_type') == gt_t and d.get('gt_idx') == gt_i
                    and d.get('pred_type') == pt_t
                    and d.get('pred_idx') == pt_i):
                return i
        return 0

    def _on_review_filter_changed(self, value):
        """Handle review status filter dropdown change."""
        prev_det = self._current_review_det()
        mapping = {"All": "all", "Reviewed": "reviewed", "Not Reviewed": "not_reviewed"}
        self._review_status_filter = mapping.get(value, "all")
        self._rebuild_review_detections()
        self._review_detection_idx = self._refind_detection(prev_det)
        if self._review_detections:
            self._zoom_to_detection()
        else:
            self._display_review_image()
        self._update_review_labels()

    def _rebuild_review_detections(self):
        """Build the filtered detection list from current matches.

        Includes reviewed detections (with their review entry attached)
        so users can revisit them. Filters by match type, class, and
        review status.
        """
        dets = []
        matches = self._review_matches
        if not matches:
            self._review_detections = dets
            return

        ftype = self._review_filter_type
        fclass = self._review_filter_class
        fstatus = self._review_status_filter  # "all", "reviewed", "not_reviewed"

        img_name = self.images[self._review_index] if self.images else ""

        def _class_ok(cid):
            if fclass == "all":
                return True
            return cid == fclass

        def _status_ok(det):
            if fstatus == "all":
                return True
            entry = self._find_reviewed_entry(det, img_name)
            if fstatus == "reviewed":
                return entry is not None
            else:  # not_reviewed
                return entry is None

        # TPs
        if ftype in ("all", "tp"):
            for gt_type, gt_idx, p_type, p_idx, iou, cid, conf in matches['tp']:
                if not _class_ok(cid):
                    continue
                bbox = self._match_bbox(gt_type, gt_idx, p_type, p_idx)
                det = {
                    'det_type': 'tp', 'class_id': cid, 'conf': conf,
                    'iou': iou,
                    'gt_type': gt_type, 'gt_idx': gt_idx,
                    'pred_type': p_type, 'pred_idx': p_idx,
                    'bbox': bbox,
                }
                if _status_ok(det):
                    dets.append(det)

        # FPs
        if ftype in ("all", "fp"):
            for p_type, p_idx, cid, conf in matches['fp']:
                if not _class_ok(cid):
                    continue
                bbox = self._match_bbox(None, None, p_type, p_idx)
                det = {
                    'det_type': 'fp', 'class_id': cid, 'conf': conf,
                    'iou': None,
                    'gt_type': None, 'gt_idx': None,
                    'pred_type': p_type, 'pred_idx': p_idx,
                    'bbox': bbox,
                }
                if _status_ok(det):
                    dets.append(det)

        # FNs
        if ftype in ("all", "fn"):
            for gt_type, gt_idx, cid in matches['fn']:
                if not _class_ok(cid):
                    continue
                bbox = self._match_bbox(gt_type, gt_idx, None, None)
                det = {
                    'det_type': 'fn', 'class_id': cid, 'conf': None,
                    'iou': None,
                    'gt_type': gt_type, 'gt_idx': gt_idx,
                    'pred_type': None, 'pred_idx': None,
                    'bbox': bbox,
                }
                if _status_ok(det):
                    dets.append(det)

        self._review_detections = dets

    def _match_bbox(self, gt_type, gt_idx, p_type, p_idx):
        """Compute bounding box for a detection in image coordinates."""
        bboxes = []
        # GT bbox
        if gt_type and gt_idx is not None:
            if gt_type == 'box' and 0 <= gt_idx < len(self._review_gt_boxes):
                b = self._review_gt_boxes[gt_idx]
                bboxes.append((b[0], b[1], b[2], b[3]))
            elif gt_type == 'polygon' and 0 <= gt_idx < len(self._review_gt_polygons):
                pts = self._review_gt_polygons[gt_idx][0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        # Pred bbox
        if p_type and p_idx is not None:
            if p_type == 'box' and 0 <= p_idx < len(self._review_pred_boxes):
                b = self._review_pred_boxes[p_idx]
                bboxes.append((b[0], b[1], b[2], b[3]))
            elif p_type == 'polygon' and 0 <= p_idx < len(self._review_pred_polygons):
                pts = self._review_pred_polygons[p_idx][0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        if not bboxes:
            return (0, 0, self._review_img_w, self._review_img_h)
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return (x1, y1, x2, y2)

    # ── Per-detection review persistence ─────────────────────────────────────

    def _det_norm_bbox(self, det, which='auto'):
        """Get normalized [cx, cy, w, h] for a detection's GT or pred bbox.

        which: 'gt', 'pred', or 'auto' (pred if available, else gt).
        Returns list [cx, cy, w, h] or None.
        """
        img_w = max(self._review_img_w, 1)
        img_h = max(self._review_img_h, 1)

        def _norm(gt_or_pred_type, gt_or_pred_idx, boxes, polygons):
            if gt_or_pred_type == 'box' and gt_or_pred_idx is not None:
                if 0 <= gt_or_pred_idx < len(boxes):
                    b = boxes[gt_or_pred_idx]
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    return [
                        round((x1 + x2) / 2 / img_w, 6),
                        round((y1 + y2) / 2 / img_h, 6),
                        round((x2 - x1) / img_w, 6),
                        round((y2 - y1) / img_h, 6)]
            elif gt_or_pred_type == 'polygon' and gt_or_pred_idx is not None:
                if 0 <= gt_or_pred_idx < len(polygons):
                    pts = polygons[gt_or_pred_idx][0] if isinstance(polygons[gt_or_pred_idx], tuple) else polygons[gt_or_pred_idx]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    return [
                        round((x1 + x2) / 2 / img_w, 6),
                        round((y1 + y2) / 2 / img_h, 6),
                        round((x2 - x1) / img_w, 6),
                        round((y2 - y1) / img_h, 6)]
            return None

        gt_bbox = _norm(det.get('gt_type'), det.get('gt_idx'),
                        self._review_gt_boxes, self._review_gt_polygons)
        pred_bbox = _norm(det.get('pred_type'), det.get('pred_idx'),
                          self._review_pred_boxes, self._review_pred_polygons)

        if which == 'gt':
            return gt_bbox
        elif which == 'pred':
            return pred_bbox
        else:  # auto
            return pred_bbox or gt_bbox

    def _build_reviewed_lookup(self, img_name):
        """Build O(1) lookup dicts for reviewed entries on an image."""
        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            self._reviewed_lookup = (img_name, {}, {})
            return
        reviewed_dets = img_data.get("detections", [])
        QUANT = 500  # 1 / TOLERANCE
        pred_map = {}  # (qcx, qcy) -> [entries]
        gt_map = {}    # (qcx, qcy) -> [entries]
        for entry in reviewed_dets:
            e_pred = entry.get("pred_bbox_norm")
            if e_pred:
                qk = (round(e_pred[0] * QUANT), round(e_pred[1] * QUANT))
                pred_map.setdefault(qk, []).append(entry)
            e_gt = entry.get("gt_bbox_norm")
            if e_gt:
                qk = (round(e_gt[0] * QUANT), round(e_gt[1] * QUANT))
                gt_map.setdefault(qk, []).append(entry)
        self._reviewed_lookup = (img_name, pred_map, gt_map)

    def _invalidate_reviewed_lookup(self):
        """Mark the reviewed-entry lookup as stale."""
        self._reviewed_lookup = ("", {}, {})

    def _find_reviewed_entry(self, det, img_name):
        """Find a matching reviewed entry for a detection.

        Uses a spatial hash for O(1) amortized lookup instead of linear scan.
        Matching strategy:
        - For TP/FP (has prediction): match by pred_bbox_norm center (within 0.002)
        - For FN (no prediction): match by gt_bbox_norm center (within 0.002)

        Returns the reviewed entry dict or None.
        """
        if not img_name:
            return None

        # Rebuild lookup if needed (different image or invalidated)
        if self._reviewed_lookup[0] != img_name:
            self._build_reviewed_lookup(img_name)

        _, pred_map, gt_map = self._reviewed_lookup

        TOLERANCE = 0.002
        QUANT = 500

        det_type = det['det_type']
        if det_type in ('tp', 'fp'):
            pred_bbox = self._det_norm_bbox(det, 'pred')
            if not pred_bbox:
                return None
            pcx, pcy = pred_bbox[0], pred_bbox[1]
            qx, qy = round(pcx * QUANT), round(pcy * QUANT)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for entry in pred_map.get((qx + dx, qy + dy), ()):
                        e_pred = entry.get("pred_bbox_norm")
                        if (e_pred and abs(e_pred[0] - pcx) < TOLERANCE
                                and abs(e_pred[1] - pcy) < TOLERANCE):
                            return entry
        else:  # fn
            gt_bbox = self._det_norm_bbox(det, 'gt')
            if not gt_bbox:
                return None
            gcx, gcy = gt_bbox[0], gt_bbox[1]
            qx, qy = round(gcx * QUANT), round(gcy * QUANT)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for entry in gt_map.get((qx + dx, qy + dy), ()):
                        e_gt = entry.get("gt_bbox_norm")
                        if (e_gt and abs(e_gt[0] - gcx) < TOLERANCE
                                and abs(e_gt[1] - gcy) < TOLERANCE):
                            return entry
        return None

    def _record_detection_action(self, det, action):
        """Record a review action for a detection and persist to disk.

        action: 'accepted', 'rejected', or 'edited'
        """
        if not self.images:
            return
        img_name = self.images[self._review_index]

        per_image = self._review_state.setdefault("image", {})
        img_data = per_image.setdefault(img_name, {"img_status": "started", "detections": []})

        # Build the reviewed entry
        class_name = self.class_names.get(det['class_id'], f"class_{det['class_id']}")
        entry = {
            "match_type": det['det_type'].upper(),
            "det_status": "reviewed",
            "action": action,
            "class_id": det['class_id'],
            "class_name": class_name,
            "gt_bbox_norm": self._det_norm_bbox(det, 'gt'),
            "pred_bbox_norm": self._det_norm_bbox(det, 'pred'),
            "iou": round(det['iou'], 4) if det.get('iou') is not None else None,
            "conf": round(det['conf'], 4) if det.get('conf') is not None else None,
        }

        # Check if already recorded (update if so)
        existing = self._find_reviewed_entry(det, img_name)
        if existing:
            # Update in place
            idx = img_data["detections"].index(existing)
            img_data["detections"][idx] = entry
        else:
            img_data["detections"].append(entry)

        # Invalidate lookup so next query rebuilds from updated data
        self._invalidate_reviewed_lookup()

        # Ensure img_status is at least "started"
        if img_data.get("img_status") == "not_started":
            img_data["img_status"] = "started"

        self._save_review_state()

    def _check_image_review_complete(self):
        """Check if all detections on current image are reviewed, mark complete."""
        if not self.images:
            return
        img_name = self.images[self._review_index]

        # Count total detections (unfiltered) and reviewed ones
        matches = self._review_matches
        if not matches:
            return

        total = len(matches.get('tp', [])) + len(matches.get('fp', [])) + len(matches.get('fn', []))
        if total == 0:
            return

        per_image = self._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return

        reviewed_count = len(img_data.get("detections", []))
        if reviewed_count >= total:
            img_data["img_status"] = "completed"
            self._save_review_state()

    def _zoom_to_detection(self):
        """Auto-zoom and center the review canvas on the current detection."""
        # Reset GT/Pred checkboxes to checked on each detection focus
        self._review_gt_var.set(True)
        self._review_show_gt = True
        self._review_pred_var.set(True)
        self._review_show_pred = True
        if not self._review_detections:
            self._display_review_image()
            return
        idx = max(0, min(self._review_detection_idx,
                         len(self._review_detections) - 1))
        det = self._review_detections[idx]
        bbox = det.get('bbox')
        if not bbox:
            self._display_review_image()
            return

        x1, y1, x2, y2 = bbox
        det_w = max(x2 - x1, 1)
        det_h = max(y2 - y1, 1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cw = self._review_canvas.winfo_width() or 800
        ch = self._review_canvas.winfo_height() or 600

        # Detection should fill ~1/3 of canvas (3x context)
        scale_x = cw / (det_w * 3)
        scale_y = ch / (det_h * 3)
        new_scale = max(0.05, min(10.0, min(scale_x, scale_y)))

        self._review_scale = new_scale
        self._review_offset_x = cw / 2 - cx * new_scale
        self._review_offset_y = ch / 2 - cy * new_scale
        self._review_cached_scale = None
        self._display_review_image()

    def _update_review_labels(self):
        """Update all navigation counters and status labels."""
        # Use main toolbar counter/total for review image nav
        n_filtered = len(self._review_filtered_images)
        if n_filtered and self._review_index in self._review_filtered_images:
            filt_pos = self._review_filtered_images.index(self._review_index) + 1
        else:
            filt_pos = self._review_index + 1 if self.images else 0
        self.counter_entry.delete(0, "end")
        self.counter_entry.insert(0, str(filt_pos))
        self.total_label.configure(text=f"/ {n_filtered}")

        # Image name
        if self.images and 0 <= self._review_index < len(self.images):
            self.image_name_label.configure(
                text=self.images[self._review_index])

        # Detection nav label (in status bar)
        n_dets = len(self._review_detections)
        det_pos = self._review_detection_idx + 1 if n_dets else 0
        det_type = ""
        det_status_text = ""
        if n_dets and self._review_detection_idx < n_dets:
            det = self._review_detections[self._review_detection_idx]
            det_type = det['det_type'].upper() + " "
            # Show per-detection review status
            img_name = self.images[self._review_index] if self.images else ""
            entry = self._find_reviewed_entry(det, img_name) if img_name else None
            if entry:
                action = entry.get("action", "reviewed")
                det_status_text = f"({action})"
            else:
                det_status_text = "(not reviewed)"
        self._review_det_label.configure(text=f"{det_type}{det_pos} / {n_dets}")
        self._review_det_status_label.configure(text=det_status_text)

        # TP/FP/FN counts (in top toolbar)
        matches = self._review_matches
        if matches:
            tp = len(matches.get('tp', []))
            fp = len(matches.get('fp', []))
            fn = len(matches.get('fn', []))
            self._review_counts_label.configure(
                text=f"TP: {tp} | FP: {fp} | FN: {fn}")
        else:
            self._review_counts_label.configure(text="TP: 0 | FP: 0 | FN: 0")

        # Update button text based on detection context
        if n_dets and self._review_detection_idx < n_dets:
            det = self._review_detections[self._review_detection_idx]
            dt = det['det_type']
            if dt == 'fn':
                self._review_accept_btn.configure(text="Keep GT (A)")
                self._review_reject_btn.configure(text="Delete GT (R)")
            elif dt == 'fp':
                self._review_accept_btn.configure(text="Add to GT (A)")
                self._review_reject_btn.configure(text="Dismiss (R)")
            else:  # tp
                self._review_accept_btn.configure(text="Confirm (A)")
                self._review_reject_btn.configure(text="Delete GT (R)")

        # Update zoom text for review tab
        self._update_status()

    # ── Review rendering ─────────────────────────────────────────────────────

    def _display_review_image(self):
        """Render image with GT and prediction overlays on review canvas."""
        c = self._review_canvas
        c.delete("all")

        if self._review_original_image is None:
            c.create_text(
                c.winfo_width() // 2, c.winfo_height() // 2,
                text="No annotations or predictions to review!",
                fill=FG_COLOR, font=(self.font_family, 14))
            return

        s = self._review_scale
        ox, oy = self._review_offset_x, self._review_offset_y
        cw = c.winfo_width() or 800
        ch = c.winfo_height() or 600

        # Visible region in image coords
        ix0 = max(0, int(-ox / s))
        iy0 = max(0, int(-oy / s))
        ix1 = min(self._review_img_w, int((cw - ox) / s) + 1)
        iy1 = min(self._review_img_h, int((ch - oy) / s) + 1)

        crop_w = ix1 - ix0
        crop_h = iy1 - iy0
        if crop_w <= 0 or crop_h <= 0:
            return

        disp_w = max(1, int(crop_w * s))
        disp_h = max(1, int(crop_h * s))

        review_cache_key = (s, ix0, iy0, ix1, iy1)
        if (self._review_cached_scale != review_cache_key
                or self._review_cached_tk_image is None):
            region = self._review_original_image.crop(
                (ix0, iy0, ix1, iy1))
            region = region.resize(
                (disp_w, disp_h), Image.Resampling.BILINEAR)
            self._review_cached_tk_image = ImageTk.PhotoImage(region)
            self._review_cached_scale = review_cache_key

        px = ox + ix0 * s
        py = oy + iy0 * s
        c.create_image(px, py, anchor="nw",
                       image=self._review_cached_tk_image)

        # Determine which detection is focused
        focused_det = None
        if (self._review_detections
                and 0 <= self._review_detection_idx
                < len(self._review_detections)):
            focused_det = self._review_detections[self._review_detection_idx]

        # Build sets of focused GT and pred indices for highlighting
        focused_gt = set()   # (type, idx)
        focused_pred = set() # (type, idx)
        if focused_det:
            if focused_det['gt_type'] is not None:
                focused_gt.add((focused_det['gt_type'], focused_det['gt_idx']))
            if focused_det['pred_type'] is not None:
                focused_pred.add(
                    (focused_det['pred_type'], focused_det['pred_idx']))

        # Build set of reviewed GT indices from ALL matches (unfiltered)
        reviewed_gt = set()  # (type, idx) — GT annotations already reviewed
        img_name = self.images[self._review_index] if self.images else ""
        if img_name and self._review_matches:
            for gt_type, gt_idx, p_type, p_idx, iou, cid, conf in \
                    self._review_matches.get('tp', []):
                det = {'det_type': 'tp', 'class_id': cid,
                       'gt_type': gt_type, 'gt_idx': gt_idx,
                       'pred_type': p_type, 'pred_idx': p_idx}
                if self._find_reviewed_entry(det, img_name):
                    reviewed_gt.add((gt_type, gt_idx))
            for gt_type, gt_idx, cid in self._review_matches.get('fn', []):
                det = {'det_type': 'fn', 'class_id': cid,
                       'gt_type': gt_type, 'gt_idx': gt_idx,
                       'pred_type': None, 'pred_idx': None}
                if self._find_reviewed_entry(det, img_name):
                    reviewed_gt.add((gt_type, gt_idx))

        # Constant line width (1.5pt ≈ 2px) regardless of zoom
        line_w = 2
        focused_line_w = 3
        label_size = 12
        PRED_COLOR = "#00BFFF"  # highlighter blue for focused prediction
        FOCUSED_GT_COLOR = "#FFD700"  # gold for focused GT annotation

        REVIEWED_STIPPLE = "gray12"  # ~12% pixel density for transparent fill

        def _halo_text(x, y, text, fill, **kw):
            """Draw text with black outline halo for readability."""
            for dx, dy in [(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),
                           (-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),
                           (0,-2),(0,-1),(0,1),(0,2),
                           (1,-2),(1,-1),(1,0),(1,1),(1,2),
                           (2,-2),(2,-1),(2,0),(2,1),(2,2)]:
                c.create_text(x + dx, y + dy, text=text,
                              fill="black", **kw)
            c.create_text(x, y, text=text, fill=fill, **kw)

        # Determine GT draw format: match prediction format.
        # If predictions are boxes → draw GT as boxes.
        # If predictions are polygons → draw GT as polygons.
        # If no predictions → use current mode.
        has_pred_boxes = bool(self._review_pred_boxes)
        has_pred_polys = bool(self._review_pred_polygons)
        if has_pred_boxes and not has_pred_polys:
            gt_draw_mode = "box"
        elif has_pred_polys and not has_pred_boxes:
            gt_draw_mode = "polygon"
        elif has_pred_boxes and has_pred_polys:
            gt_draw_mode = "both"
        else:
            gt_draw_mode = self.mode  # no predictions: use mode button

        # ── Draw GT annotations (format matched to predictions) ──
        if self._review_show_gt:
            # Focused GT is deferred to draw last (z-order: always on top)
            _deferred_focused_gt = []

            # Draw GT boxes (or polygon GT converted to bboxes)
            if gt_draw_mode in ("box", "both"):
                for i, (x1, y1, x2, y2, cid) in enumerate(
                        self._review_gt_boxes):
                    is_focused = ('box', i) in focused_gt
                    if is_focused:
                        _deferred_focused_gt.append(
                            ('rect', x1, y1, x2, y2, cid))
                        continue
                    # Viewport culling
                    if x2 < ix0 or x1 > ix1 or y2 < iy0 or y1 > iy1:
                        continue
                    color = self._get_class_color(cid)
                    is_reviewed = ('box', i) in reviewed_gt
                    cx1, cy1 = self._review_image_to_canvas(x1, y1)
                    cx2, cy2 = self._review_image_to_canvas(x2, y2)
                    if is_reviewed:
                        c.create_rectangle(cx1, cy1, cx2, cy2,
                                           outline=color, width=line_w,
                                           fill=color, stipple=REVIEWED_STIPPLE)
                    else:
                        c.create_rectangle(cx1, cy1, cx2, cy2,
                                           outline=color, width=line_w,
                                           fill="")
                # Also draw polygon GT as bounding boxes
                if gt_draw_mode == "box":
                    for i, (pts, cid) in enumerate(
                            self._review_gt_polygons):
                        is_focused = ('polygon', i) in focused_gt
                        if is_focused:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            _deferred_focused_gt.append(
                                ('rect', min(xs), min(ys),
                                 max(xs), max(ys), cid))
                            continue
                        # Viewport culling
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        if max(xs) < ix0 or min(xs) > ix1 or max(ys) < iy0 or min(ys) > iy1:
                            continue
                        color = self._get_class_color(cid)
                        is_reviewed = ('polygon', i) in reviewed_gt
                        bx1, by1 = min(xs), min(ys)
                        bx2, by2 = max(xs), max(ys)
                        cx1, cy1 = self._review_image_to_canvas(bx1, by1)
                        cx2, cy2 = self._review_image_to_canvas(bx2, by2)
                        if is_reviewed:
                            c.create_rectangle(cx1, cy1, cx2, cy2,
                                               outline=color, width=line_w,
                                               fill=color, stipple=REVIEWED_STIPPLE)
                        else:
                            c.create_rectangle(cx1, cy1, cx2, cy2,
                                               outline=color, width=line_w,
                                               fill="")

            # Draw GT polygons (or box GT converted to rectangles as polygons)
            if gt_draw_mode in ("polygon", "both"):
                for i, (pts, cid) in enumerate(self._review_gt_polygons):
                    is_focused = ('polygon', i) in focused_gt
                    if is_focused:
                        _deferred_focused_gt.append(('poly', pts, cid))
                        continue
                    # Viewport culling
                    pxs = [p[0] for p in pts]
                    pys = [p[1] for p in pts]
                    if max(pxs) < ix0 or min(pxs) > ix1 or max(pys) < iy0 or min(pys) > iy1:
                        continue
                    color = self._get_class_color(cid)
                    is_reviewed = ('polygon', i) in reviewed_gt
                    canvas_pts = []
                    for px_pt, py_pt in pts:
                        cx_p, cy_p = self._review_image_to_canvas(
                            px_pt, py_pt)
                        canvas_pts.extend([cx_p, cy_p])
                    if len(canvas_pts) >= 6:
                        if is_reviewed:
                            c.create_polygon(*canvas_pts, outline=color,
                                             width=line_w, fill=color,
                                             stipple=REVIEWED_STIPPLE)
                        else:
                            c.create_polygon(*canvas_pts, outline=color,
                                             width=line_w, fill="")
                # Also draw box GT as polygons
                if gt_draw_mode == "polygon":
                    for i, (x1, y1, x2, y2, cid) in enumerate(
                            self._review_gt_boxes):
                        is_focused = ('box', i) in focused_gt
                        if is_focused:
                            _deferred_focused_gt.append(
                                ('poly', [(x1, y1), (x2, y1),
                                          (x2, y2), (x1, y2)], cid))
                            continue
                        # Viewport culling
                        if x2 < ix0 or x1 > ix1 or y2 < iy0 or y1 > iy1:
                            continue
                        color = self._get_class_color(cid)
                        is_reviewed = ('box', i) in reviewed_gt
                        rect_pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        canvas_pts = []
                        for px_pt, py_pt in rect_pts:
                            cx_p, cy_p = self._review_image_to_canvas(
                                px_pt, py_pt)
                            canvas_pts.extend([cx_p, cy_p])
                        if is_reviewed:
                            c.create_polygon(*canvas_pts, outline=color,
                                             width=line_w, fill=color,
                                             stipple=REVIEWED_STIPPLE)
                        else:
                            c.create_polygon(*canvas_pts, outline=color,
                                             width=line_w, fill="")

            # ── Draw focused GT last (always on top) ──
            for _dfg in _deferred_focused_gt:
                if _dfg[0] == 'rect':
                    _, fx1, fy1, fx2, fy2, fcid = _dfg
                    cx1, cy1 = self._review_image_to_canvas(fx1, fy1)
                    cx2, cy2 = self._review_image_to_canvas(fx2, fy2)
                    c.create_rectangle(cx1, cy1, cx2, cy2,
                                       outline=FOCUSED_GT_COLOR,
                                       fill="", width=focused_line_w)
                    name = self.class_names.get(fcid, str(fcid))
                    _halo_text(cx1 + 2, cy1 - 2, anchor="sw",
                               text=f"GT {fcid}: {name}",
                               fill=FOCUSED_GT_COLOR,
                               font=(self.font_family,
                                     label_size, "bold"))
                elif _dfg[0] == 'poly':
                    _, fpts, fcid = _dfg
                    canvas_pts = []
                    for px_pt, py_pt in fpts:
                        cx_p, cy_p = self._review_image_to_canvas(
                            px_pt, py_pt)
                        canvas_pts.extend([cx_p, cy_p])
                    if len(canvas_pts) >= 6:
                        c.create_polygon(*canvas_pts,
                                         outline=FOCUSED_GT_COLOR,
                                         fill="", width=focused_line_w)
                    if fpts:
                        lx, ly = self._review_image_to_canvas(*fpts[0])
                        name = self.class_names.get(fcid, str(fcid))
                        _halo_text(lx + 2, ly - 2, anchor="sw",
                                   text=f"GT {fcid}: {name}",
                                   fill=FOCUSED_GT_COLOR,
                                   font=(self.font_family,
                                         label_size, "bold"))

        # ── Draw ONLY the focused prediction (blue) ──
        if self._review_show_pred and focused_det:
            p_type = focused_det.get('pred_type')
            p_idx = focused_det.get('pred_idx')
            if p_type == 'box' and p_idx is not None:
                if 0 <= p_idx < len(self._review_pred_boxes):
                    bx1, by1, bx2, by2, cid, conf = \
                        self._review_pred_boxes[p_idx]
                    cx1, cy1 = self._review_image_to_canvas(bx1, by1)
                    cx2, cy2 = self._review_image_to_canvas(bx2, by2)
                    c.create_rectangle(cx1, cy1, cx2, cy2,
                                       outline=PRED_COLOR, width=line_w)
                    name = self.class_names.get(cid, str(cid))
                    _halo_text(cx1 + 2, cy2 + 2, anchor="nw",
                               text=f"Pred {cid}: {name} ({conf:.2f})",
                               fill=PRED_COLOR,
                               font=(self.font_family, label_size, "bold"))
            elif p_type == 'polygon' and p_idx is not None:
                if 0 <= p_idx < len(self._review_pred_polygons):
                    pts, cid, conf = self._review_pred_polygons[p_idx]
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
                        c.create_polygon(*canvas_pts, outline=PRED_COLOR,
                                         fill="", width=line_w)
                    if pts:
                        name = self.class_names.get(cid, str(cid))
                        _halo_text(min_cx + 2, max_cy + 2, anchor="nw",
                                   text=f"Pred {cid}: {name} ({conf:.2f})",
                                   fill=PRED_COLOR,
                                   font=(self.font_family,
                                         label_size, "bold"))

        # ── Detection type badge (upper right, semi-transparent bg) ──
        if focused_det and (self._review_show_gt or self._review_show_pred):
            dtype = focused_det['det_type'].upper()
            badge_colors = {'TP': '#4CAF50', 'FP': '#EF5350', 'FN': '#FFA726'}
            badge_color = badge_colors.get(dtype, FG_COLOR)
            badge_text = f"[{dtype}]"
            bfnt = tkFont.Font(family=self.font_family, size=14,
                               weight="bold")
            tw = bfnt.measure(badge_text)
            th = bfnt.metrics("linespace")
            bx = cw - tw - 20
            by = 10
            c.create_rectangle(bx - 6, by - 2, bx + tw + 6, by + th + 4,
                               fill="#1A1A1A", outline="#444444", width=1)
            c.create_text(bx, by + 2, anchor="nw",
                          text=badge_text, fill=badge_color,
                          font=(self.font_family, 14, "bold"))

        # ── Review help overlay ──
        if self._review_show_help:
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

    # ── Review navigation ────────────────────────────────────────────────────

    def _review_next_image(self, reset_filters=True):
        if not self.images:
            return
        if reset_filters:
            # Reset filters to 'all' on manual image change
            self._review_filter_type = "all"
            self._review_type_var.set("All")
            self._review_status_filter = "all"
            self._review_filter_var.set("All")
            self._review_filter_class = "all"
            self._review_class_var.set("All")
        filt = self._review_filtered_images
        if filt:
            try:
                pos = filt.index(self._review_index)
                self._review_index = filt[(pos + 1) % len(filt)]
            except ValueError:
                self._review_index = filt[0]
        else:
            self._review_index = (self._review_index + 1) % len(self.images)
        self._review_original_image = None
        self._review_cached_scale = None
        self._review_cached_tk_image = None
        self._review_load_image()

    def _review_prev_image(self, reset_filters=True):
        if not self.images:
            return
        if reset_filters:
            # Reset filters to 'all' on manual image change
            self._review_filter_type = "all"
            self._review_type_var.set("All")
            self._review_status_filter = "all"
            self._review_filter_var.set("All")
            self._review_filter_class = "all"
            self._review_class_var.set("All")
        filt = self._review_filtered_images
        if filt:
            try:
                pos = filt.index(self._review_index)
                self._review_index = filt[(pos - 1) % len(filt)]
            except ValueError:
                self._review_index = filt[-1]
        else:
            self._review_index = (self._review_index - 1) % len(self.images)
        self._review_original_image = None
        self._review_cached_scale = None
        self._review_cached_tk_image = None
        self._review_load_image()

    def _review_next_detection(self):
        if not self._review_detections:
            return
        self._review_detection_idx = (
            (self._review_detection_idx + 1) % len(self._review_detections))
        self._zoom_to_detection()
        self._update_review_labels()

    def _review_prev_detection(self):
        if not self._review_detections:
            return
        self._review_detection_idx = (
            (self._review_detection_idx - 1) % len(self._review_detections))
        self._zoom_to_detection()
        self._update_review_labels()

    # ── Review filter callbacks ──────────────────────────────────────────────

    def _on_review_type_changed(self, choice):
        prev_det = self._current_review_det()
        self._review_filter_type = choice.lower()
        self._rebuild_review_detections()
        self._review_detection_idx = self._refind_detection(prev_det)
        if self._review_detections:
            self._zoom_to_detection()
        else:
            self._display_review_image()
        self._update_review_labels()

    def _on_review_class_changed(self, choice):
        prev_det = self._current_review_det()
        if choice == "All":
            self._review_filter_class = "all"
        else:
            try:
                self._review_filter_class = int(choice.split(":")[0])
            except (ValueError, IndexError):
                self._review_filter_class = "all"
        self._rebuild_review_detections()
        self._review_detection_idx = self._refind_detection(prev_det)
        if self._review_detections:
            self._zoom_to_detection()
        else:
            self._display_review_image()
        self._update_review_labels()

    # ── Review actions ───────────────────────────────────────────────────────

    def _review_accept(self):
        """Accept the current detection.

        TP: Confirm match is correct — step to next.
        FP: Switch to Annotate to draw annotation with
            prediction shown as blue reference overlay.
        FN: Keep GT as-is (model just missed it) — step to next.
        """
        if not self._review_detections:
            return
        det = self._review_detections[self._review_detection_idx]

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
        if not self._review_detections:
            return
        det = self._review_detections[self._review_detection_idx]

        if det['det_type'] in ('fn', 'tp'):
            # Record action before GT deletion changes the list
            self._record_detection_action(det, "rejected")
            # Backup originals before first destructive edit
            self._backup_original_labels()
            # Delete GT annotation from disk
            gt_type = det['gt_type']
            gt_idx = det['gt_idx']
            if gt_type == 'box' and gt_idx is not None:
                if 0 <= gt_idx < len(self._review_gt_boxes):
                    self._review_gt_boxes.pop(gt_idx)
                    self._review_save_gt()
            elif gt_type == 'polygon' and gt_idx is not None:
                if 0 <= gt_idx < len(self._review_gt_polygons):
                    self._review_gt_polygons.pop(gt_idx)
                    self._review_save_gt()
            self._review_recompute_and_advance()
        else:
            # FP reject = dismiss, step to next
            self._review_step_next(action="rejected")

    def _review_edit(self):
        """Switch to Annotate tab for editing, with prediction reference.

        All types: switch to Annotate zoomed to the detection,
        with the prediction (if any) shown as a dashed blue reference.
        """
        if not self._review_detections:
            return
        det = self._review_detections[self._review_detection_idx]
        self._switch_to_annotate_for_review(det)

    def _switch_to_annotate_for_review(self, det):
        """Switch to Annotate tab from Review, showing prediction reference."""
        if not self.images:
            return
        # Capture review zoom/offset
        rev_scale = self._review_scale
        rev_ox = self._review_offset_x
        rev_oy = self._review_offset_y

        # Build prediction reference overlay data
        pred_ref = None
        p_type = det.get('pred_type')
        p_idx = det.get('pred_idx')
        if p_type == 'box' and p_idx is not None:
            if 0 <= p_idx < len(self._review_pred_boxes):
                b = self._review_pred_boxes[p_idx]
                pred_ref = {
                    'type': 'box',
                    'coords': (b[0], b[1], b[2], b[3]),
                    'class_id': b[4], 'conf': b[5]}
        elif p_type == 'polygon' and p_idx is not None:
            if 0 <= p_idx < len(self._review_pred_polygons):
                pts, cid, conf = self._review_pred_polygons[p_idx]
                pred_ref = {
                    'type': 'polygon',
                    'coords': list(pts),
                    'class_id': cid, 'conf': conf}

        self._annotate_pred_reference = pred_ref
        self._review_return_pending = True
        self._review_editing_det = det

        # Navigate annotate tab to same image (defer display to avoid flash)
        self._defer_display = True
        self.index = self._review_index
        self.load_image()
        self._defer_display = False

        # Sync zoom/position from review to annotate
        self.scale = rev_scale
        self.offset_x = rev_ox
        self.offset_y = rev_oy
        self.zoom_index = self._nearest_zoom_index(rev_scale)
        self._cached_scale = None

        # Determine appropriate mode: prefer polygon if polygon labels exist
        gt_type = det.get('gt_type')
        gt_idx = det.get('gt_idx')
        # Always default to polygon when polygon labels are present
        if self.polygons:
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
            if self.mode != 'polygon':
                self.mode = 'polygon'
                self.mode_btn.configure(text="Mode: Polygon \u2b21")
                self.stream_btn.pack(side="left", padx=(0, 4))
                self.snap_btn.pack(side="left", padx=(0, 4))
            if gt_idx is not None and 0 <= gt_idx < len(self.polygons):
                self._selected_polygon_idx = gt_idx
        elif target_mode == 'box':
            if self.mode != 'box':
                self.mode = 'box'
                self.mode_btn.configure(text="Mode: Box \u25ad")
                self.current_polygon = []
                self._selected_polygon_idx = None
                self.stream_btn.pack_forget()
                self.snap_btn.pack_forget()

        # Switch tab and display with correct zoom
        self.tabview.set("Annotate")
        self._on_tab_changed()

    def _review_confirm_dialog(self):
        """Show confirmation dialog after annotation while in review flow."""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Confirm GT Edits")
        dialog.geometry("360x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(fg_color=BG_COLOR)

        # Center on parent
        dialog.update_idletasks()
        px = self.root.winfo_x() + (self.root.winfo_width() - 360) // 2
        py = self.root.winfo_y() + (self.root.winfo_height() - 150) // 2
        dialog.geometry(f"+{px}+{py}")

        ctk.CTkLabel(
            dialog,
            text="Save this annotation to the GT dataset?",
            font=(self.font_family, 13, "bold"),
            text_color=FG_COLOR,
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            dialog,
            text="Accept saves the edits and returns to review.\n"
                 "Redo undoes the last edits so you can redo.",
            font=(self.font_family, 11),
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
            self.undo_last()

        ctk.CTkButton(
            btn_frame, text="\u2713 Accept", width=120,
            fg_color=SI_GREEN, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12, "bold"),
            command=on_accept,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            btn_frame, text="\u21ba Redo", width=120,
            fg_color=SI_PERSIMMON, hover_color="#F0A77B",
            text_color=FG_COLOR, font=(self.font_family, 12, "bold"),
            command=on_redo,
        ).pack(side="left")

    def _save_and_return_to_review(self):
        """Save current annotations and return to Review tab."""
        self.save_annotations()
        # Record the action for the detection that was being edited
        if hasattr(self, '_review_editing_det') and self._review_editing_det:
            self._record_detection_action(self._review_editing_det, "edited")
            self._review_editing_det = None
        self._annotate_pred_reference = None
        self._review_return_pending = False
        # Switch back to Review and recompute+advance (GT was modified)
        self._review_recompute_on_return = True
        self.tabview.set("Review")
        self._on_tab_changed()

    def _review_advance_or_switch_type(self):
        """Advance to next image, or switch match-type filter if unreviewed dets remain.

        If the current filter is not 'all', checks other match types for
        unreviewed detections on the same image before advancing.
        """
        if self._review_filter_type != "all" and self._review_matches:
            img_name = self.images[self._review_index] if self.images else ""
            type_order = ["tp", "fp", "fn"]
            for mtype in type_order:
                if mtype == self._review_filter_type:
                    continue
                entries = self._review_matches.get(mtype, [])
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
                    if not self._find_reviewed_entry(det, img_name):
                        # Found unreviewed det in another type — switch filter
                        self._review_filter_type = mtype
                        self._review_type_var.set(mtype.upper())
                        self._rebuild_review_detections()
                        self._review_detection_idx = 0
                        # Jump to first unreviewed in new list
                        for i, d in enumerate(self._review_detections):
                            if not self._find_reviewed_entry(d, img_name):
                                self._review_detection_idx = i
                                break
                        self._zoom_to_detection()
                        self._update_review_labels()
                        return
        # Also check with "all" filter to catch any missed detections
        if self._review_filter_type != "all" and self._review_matches:
            img_name = self.images[self._review_index] if self.images else ""
            saved_type = self._review_filter_type
            self._review_filter_type = "all"
            self._rebuild_review_detections()
            has_unreviewed = False
            for d in self._review_detections:
                if not self._find_reviewed_entry(d, img_name):
                    has_unreviewed = True
                    break
            if has_unreviewed:
                self._review_type_var.set("All")
                self._review_detection_idx = 0
                for i, d in enumerate(self._review_detections):
                    if not self._find_reviewed_entry(d, img_name):
                        self._review_detection_idx = i
                        break
                self._zoom_to_detection()
                self._update_review_labels()
                return
            # Restore and advance
            self._review_filter_type = saved_type
            self._rebuild_review_detections()
        self._review_next_image(reset_filters=False)

    def _review_step_next(self, action="accepted"):
        """Record action and advance to the next unreviewed detection.

        Used after non-modifying actions (TP accept, FN accept, FP reject).
        The action parameter is stored in review_state for tracking.
        """
        if not self._review_detections:
            self._check_image_review_complete()
            self._review_advance_or_switch_type()
            return
        # Record the action for the current detection
        det = self._review_detections[self._review_detection_idx]
        self._record_detection_action(det, action)
        # Advance to next unreviewed detection (skip already-reviewed ones)
        img_name = self.images[self._review_index] if self.images else ""
        start = self._review_detection_idx
        n = len(self._review_detections)
        for offset in range(1, n + 1):
            next_idx = start + offset
            if next_idx >= n:
                # Reached end of list — check if all reviewed
                self._check_image_review_complete()
                self._review_advance_or_switch_type()
                return
            next_det = self._review_detections[next_idx]
            if not self._find_reviewed_entry(next_det, img_name):
                # Found next unreviewed detection
                self._review_detection_idx = next_idx
                self._zoom_to_detection()
                self._update_review_labels()
                return

    def _review_reload_gt_and_advance(self):
        """Reload GT from disk after annotate edit, recompute, and advance."""
        if not self.images:
            return
        img_name = self.images[self._review_index]
        stem = os.path.splitext(img_name)[0]

        # Reload GT boxes
        self._review_gt_boxes = []
        detect_path = os.path.join(self.detect_dir, f"{stem}.txt")
        try:
            boxes, _ = parse_detect_labels(
                detect_path, self._review_img_w, self._review_img_h)
            self._review_gt_boxes.extend(boxes)
        except Exception:
            pass

        # Reload GT polygons
        self._review_gt_polygons = []
        segment_path = os.path.join(self.segment_dir, f"{stem}.txt")
        try:
            polys, _ = parse_segment_labels(
                segment_path, self._review_img_w, self._review_img_h)
            self._review_gt_polygons.extend(polys)
        except Exception:
            pass

        self._review_recompute_and_advance()

    def _review_recompute_and_advance(self):
        """Recompute matches after GT modification and advance.

        Used after modifying actions (FP accept/add, TP/FN reject/delete).
        """
        self._review_matches = self._compute_matches(
            self._review_gt_boxes, self._review_gt_polygons,
            self._review_pred_boxes, self._review_pred_polygons,
            iou_threshold=REVIEW_IOU_THRESHOLD,
            conf_threshold=REVIEW_CONF_THRESHOLD)
        self._rebuild_review_detections()

        if not self._review_detections:
            self._check_image_review_complete()
            self._review_advance_or_switch_type()
            return

        # Try to find next unreviewed detection from current position
        img_name = self.images[self._review_index] if self.images else ""
        found = False
        for i in range(len(self._review_detections)):
            if not self._find_reviewed_entry(self._review_detections[i], img_name):
                self._review_detection_idx = i
                found = True
                break
        if not found:
            # All reviewed in current filter after recompute
            self._check_image_review_complete()
            self._review_advance_or_switch_type()
            return
        self._zoom_to_detection()
        self._update_review_labels()

    def _backup_original_labels(self):
        """Copy all label files to .original/ on first review session per folder.

        Creates labels/detect/.original/ and labels/segment/.original/
        with copies of all label .txt files. Only runs once per folder
        (tracked by 'labels_backed_up' flag in review_stats.json).
        """
        if self._review_state.get("labels_backed_up"):
            return
        for label_dir in (self.detect_dir, self.segment_dir):
            if not os.path.isdir(label_dir):
                continue
            backup_dir = os.path.join(label_dir, ".original")
            if os.path.isdir(backup_dir):
                # Already backed up from a previous session
                continue
            txt_files = [f for f in os.listdir(label_dir)
                         if f.endswith(".txt") and os.path.isfile(
                             os.path.join(label_dir, f))]
            if not txt_files:
                continue
            os.makedirs(backup_dir, exist_ok=True)
            for fname in txt_files:
                src = os.path.join(label_dir, fname)
                dst = os.path.join(backup_dir, fname)
                shutil.copy2(src, dst)
        self._review_state["labels_backed_up"] = True
        self._save_review_state()

    def _review_save_gt(self):
        """Write current review GT boxes/polygons back to label files."""
        if not self.images:
            return
        img_name = self.images[self._review_index]
        stem = os.path.splitext(img_name)[0]

        # Save detect labels
        detect_path = os.path.join(self.detect_dir, f"{stem}.txt")
        try:
            write_detect_labels(
                detect_path, self._review_gt_boxes,
                self._review_img_w, self._review_img_h)
        except Exception as e:
            print(f"Warning: Could not save detect labels: {e}")

        # Save segment labels
        segment_path = os.path.join(self.segment_dir, f"{stem}.txt")
        try:
            write_segment_labels(
                segment_path, self._review_gt_polygons,
                self._review_img_w, self._review_img_h)
        except Exception as e:
            print(f"Warning: Could not save segment labels: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    #  Navigation
    # ──────────────────────────────────────────────────────────────────────────
    def next_image(self, event=None):
        self._record_image_time()
        self.save_annotations()
        self._save_stats()
        if self._active_filter != "all" and self._filtered_indices:
            # Jump to next image matching current filter
            for idx in self._filtered_indices:
                if idx > self.index:
                    self.index = idx
                    self.load_image()
                    return
            # Wrap to first filtered image
            self.index = self._filtered_indices[0]
            self.load_image()
            return
        self.index += 1
        if self.index >= len(self.images):
            self.index = 0
        self.load_image()

    def prev_image(self, event=None):
        self._record_image_time()
        self.save_annotations()
        self._save_stats()
        if self._active_filter != "all" and self._filtered_indices:
            for idx in reversed(self._filtered_indices):
                if idx < self.index:
                    self.index = idx
                    self.load_image()
                    return
            # Wrap to last filtered image
            self.index = self._filtered_indices[-1]
            self.load_image()
            return
        self.index -= 1
        if self.index < 0:
            self.index = len(self.images) - 1
        self.load_image()

    # ──────────────────────────────────────────────────────────────────────────
    #  Undo / Redo
    # ──────────────────────────────────────────────────────────────────────────
    def _push_undo(self):
        """Snapshot current annotation state before a mutation."""
        snapshot = (
            list(self.boxes),
            list(self.polygons),
            self._selected_polygon_idx,
        )
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)

    def undo_last(self, event=None):
        # While drawing, undo one vertex at a time
        if self.current_polygon:
            pt = self.current_polygon.pop()
            self._vertex_redo_stack.append(pt)
            if self.current_polygon:
                # Still has vertices — just redraw
                self.display_image()
                return
            # Popped the last vertex (cancelled polygon start) — fall
            # through to snapshot restore so the previous real action
            # (e.g. a delete) is undone in the same keystroke.
            self.display_image()
            if not self._undo_stack:
                return
        # Otherwise restore from snapshot stack
        if not self._undo_stack:
            return
        # Push current state to redo
        redo_snapshot = (
            list(self.boxes),
            list(self.polygons),
            self._selected_polygon_idx,
        )
        self._redo_stack.append(redo_snapshot)
        # Restore previous state
        boxes, polygons, sel_idx = self._undo_stack.pop()
        self.boxes = boxes
        self.polygons = polygons
        self._invalidate_poly_bboxes()
        self._selected_polygon_idx = sel_idx
        self._clear_drag_state()
        self.display_image()
        self.update_title()

    def redo_last(self, event=None):
        # While drawing, redo vertex
        if self.current_polygon and self._vertex_redo_stack:
            pt = self._vertex_redo_stack.pop()
            self.current_polygon.append(pt)
            self.display_image()
            return
        # Otherwise restore from redo stack
        if not self._redo_stack:
            return
        undo_snapshot = (
            list(self.boxes),
            list(self.polygons),
            self._selected_polygon_idx,
        )
        self._undo_stack.append(undo_snapshot)
        boxes, polygons, sel_idx = self._redo_stack.pop()
        self.boxes = boxes
        self.polygons = polygons
        self._invalidate_poly_bboxes()
        self._selected_polygon_idx = sel_idx
        self._clear_drag_state()
        self.display_image()
        self.update_title()

    # ──────────────────────────────────────────────────────────────────────────
    #  Save annotations
    # ──────────────────────────────────────────────────────────────────────────
    def save_annotations(self):
        if not self.images or not self.labels_dir or not self.detect_dir or not self.segment_dir:
            return
        print(f"[YoloLabeler] Saving annotations for {self.images[self.index]} "
              f"({len(self.boxes)} boxes, {len(self.polygons)} polygons)")
        os.makedirs(self.detect_dir, exist_ok=True)
        os.makedirs(self.segment_dir, exist_ok=True)

        stem = os.path.splitext(self.images[self.index])[0]
        detect_path = os.path.join(self.detect_dir, f"{stem}.txt")
        segment_path = os.path.join(self.segment_dir, f"{stem}.txt")
        img_w = self.img_width
        img_h = self.img_height
        if img_w <= 0 or img_h <= 0:
            print(f"Warning: Invalid image dimensions ({img_w}x{img_h}), skipping save")
            return

        try:
            write_detect_labels(detect_path, self.boxes, img_w, img_h)
            write_segment_labels(segment_path, self.polygons, img_w, img_h)
        except OSError as e:
            print(f"Warning: Could not save annotations: {e}")


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
