"""
YoloLabeler v2 — Image Annotation Tool for YOLO Training

CustomTkinter dark-theme UI inspired by Roboflow/CVAT.  Standard dark gray
background (#1E1E1E) with light gray text (#E0E0E0) and SI Green accent.

Features:
- Draw bounding boxes (detection) or polygons (instance segmentation)
- Toggle between Box and Polygon mode via toolbar button or 'm' key
- Save annotations in YOLO format — labels/detect/ and labels/segment/ dirs
- Multi-class via editable dropdown (type new name + Enter to create)
- Custom class colors via color picker
- Ctrl+Scroll to zoom, Scroll to pan, Shift+Scroll to pan horizontally
- Middle-click drag to pan
- Fit to width ('f'), fit to height ('g')
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
import re
import sys
import csv
import json
import math
import time
import getpass
import datetime
import contextlib
import copy
import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ExifTags

import customtkinter as ctk

# ── Paths ──────────────────────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# ── Constants ──────────────────────────────────────────────────────────────────
VERTEX_HANDLE_RADIUS = 4
STREAM_MIN_DISTANCE = 6   # min image-pixel distance between streamed vertices
SNAP_RADIUS = 15           # canvas-pixel radius for vertex/edge snapping

DEFAULT_CLASS_NAMES = {0: "catkin", 1: "bud"}

# ── Dark Theme Palette ─────────────────────────────────────────────────────────
BG_COLOR = "#1E1E1E"       # dark gray background
FG_COLOR = "#E0E0E0"       # light gray text
ACCENT = "#507754"          # SI green — buttons, highlights
ACCENT_HOVER = "#608864"    # slightly lighter green for hovers
CANVAS_BG = "#2D2D2D"      # canvas background
ENTRY_BG = "#2A2A2A"       # entry/combo background
BORDER_COLOR = "#3A3A3A"   # subtle borders

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


def _load_custom_fonts():
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
            return True
        except Exception:
            return False
    return False


def _get_font_family():
    return "Archivo"


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


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


# ══════════════════════════════════════════════════════════════════════════════
#  YoloLabeler  (v2)
# ══════════════════════════════════════════════════════════════════════════════

class YoloLabeler:
    def __init__(self, root, image_folder=None, class_names=None):
        self.root = root
        self.image_folder = image_folder
        self.class_names = (dict(class_names) if class_names
                            else dict(DEFAULT_CLASS_NAMES))
        self.class_colors = {}
        self.images = []
        self.labels_dir = None
        self.detect_dir = None
        self.segment_dir = None
        self.img_width = 0
        self.img_height = 0
        self.original_image = None

        self.mode = "box"

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
        self.show_help = True

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
        self._resize_after_id = None
        self._fast_resample = False
        self.index = 0

        # Time tracking
        self._image_start_time = None
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

        # SI logo image ref (prevent GC)
        self._logo_image = None

        # ── Build GUI ──
        _load_custom_fonts()
        self.font_family = _get_font_family()
        self._build_toolbar()
        self.canvas = tk.Canvas(root, cursor="cross", bg=CANVAS_BG,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._build_status_bar()
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

        # ── Logo ──
        self._load_logo(inner)

        # ── Open Folder (leftmost action) ──
        self.open_btn = ctk.CTkButton(
            inner, text="\U0001f4c2 Open Folder", width=120,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12),
            command=self._open_folder)
        self.open_btn.pack(side="left", padx=(8, 8))

        self._toolbar_sep(inner)

        # ── Class dropdown (editable) ──
        ctk.CTkLabel(inner, text="Class:", font=(self.font_family, 12),
                     text_color=FG_COLOR).pack(side="left", padx=(4, 4))
        self.class_var = tk.StringVar()
        self.class_dropdown = ctk.CTkComboBox(
            inner, variable=self.class_var, width=180,
            font=(self.font_family, 11),
            dropdown_font=(self.font_family, 11),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, dropdown_fg_color=BG_COLOR,
            dropdown_text_color=FG_COLOR,
            dropdown_hover_color=ACCENT,
            state="normal",
            command=self._on_class_selected)
        self.class_dropdown.pack(side="left", padx=(0, 4))
        self.class_dropdown.bind("<Return>", self._on_class_enter)
        # Also bind Return on the internal entry widget of CTkComboBox
        try:
            self.class_dropdown._entry.bind("<Return>", self._on_class_enter)
        except AttributeError:
            pass
        self._refresh_class_dropdown()

        # ── Add Class button ──
        self.add_class_btn = ctk.CTkButton(
            inner, text="+ Class", width=70,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._add_class_dialog)
        self.add_class_btn.pack(side="left", padx=(2, 4))

        # Color swatch
        self.color_btn = tk.Button(
            inner, text="  ", width=2, relief="flat", borderwidth=1,
            command=self._pick_class_color,
            bg=self._get_class_color(self.active_class),
            activebackground=self._get_class_color(self.active_class))
        self.color_btn.pack(side="left", padx=(2, 8))

        self._toolbar_sep(inner)

        # ── Complete checkbox ──
        self._complete_var = tk.BooleanVar(value=False)
        self.complete_cb = ctk.CTkCheckBox(
            inner, text="Complete", variable=self._complete_var,
            font=(self.font_family, 11), text_color=FG_COLOR,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            border_color=BORDER_COLOR,
            command=self._on_complete_toggled)
        self.complete_cb.pack(side="left", padx=(4, 4))

        # ── Filter dropdown ──
        ctk.CTkLabel(inner, text="Filter:", font=(self.font_family, 11),
                     text_color=FG_COLOR).pack(side="left", padx=(8, 2))
        self.filter_var = tk.StringVar(value="All")
        self.filter_dropdown = ctk.CTkComboBox(
            inner, variable=self.filter_var, width=130,
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

        self._toolbar_sep(inner)

        # ── Right side: navigation ──
        self.next_btn = ctk.CTkButton(
            inner, text="Next \u25b6", width=70,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12, "bold"),
            command=self.next_image)
        self.next_btn.pack(side="right", padx=(2, 4))

        self.total_label = ctk.CTkLabel(
            inner, text="/ 0", font=(self.font_family, 12),
            text_color=FG_COLOR)
        self.total_label.pack(side="right", padx=(2, 4))

        self.counter_entry = ctk.CTkEntry(
            inner, width=55, font=(self.font_family, 12),
            fg_color=ENTRY_BG, border_color=BORDER_COLOR,
            text_color=FG_COLOR, justify="center")
        self.counter_entry.pack(side="right", padx=(2, 0))
        self.counter_entry.bind("<Return>", self._on_counter_enter)
        self.counter_entry.bind("<FocusOut>", self._on_counter_focus_out)

        self.prev_btn = ctk.CTkButton(
            inner, text="\u25c0 Prev", width=70,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 12, "bold"),
            command=self.prev_image)
        self.prev_btn.pack(side="right", padx=(4, 2))

        # ── Image name label (just left of Prev) ──
        self.image_name_label = ctk.CTkLabel(
            inner, text="", font=(self.font_family, 11),
            text_color="#AAAAAA")
        self.image_name_label.pack(side="right", padx=(8, 6))

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

        # ── Left side: Mode, Stream, Snap, Fit ──
        self.mode_btn = ctk.CTkButton(
            si, text="Mode: Box \u25ad", width=120,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_mode)
        self.mode_btn.pack(side="left", padx=(0, 4))

        self.stream_btn = ctk.CTkButton(
            si, text="Stream: Off", width=95,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_stream)
        self.stream_btn.pack(side="left", padx=(0, 4))

        self.snap_btn = ctk.CTkButton(
            si, text="Snap: Off", width=80,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._toggle_snap)
        self.snap_btn.pack(side="left", padx=(0, 4))

        self._status_sep(si)

        self.fit_w_btn = ctk.CTkButton(
            si, text="Fit W", width=55,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._fit_to_width)
        self.fit_w_btn.pack(side="left", padx=(0, 4))

        self.fit_h_btn = ctk.CTkButton(
            si, text="Fit H", width=55,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=FG_COLOR, font=(self.font_family, 11),
            command=self._fit_to_height)
        self.fit_h_btn.pack(side="left", padx=(0, 4))

        # ── Right side: User, Time, Zoom ──
        self.status_user = ctk.CTkLabel(
            si, text=f"User: {self._current_user}",
            font=(self.font_family, 11), text_color=FG_COLOR)
        self.status_user.pack(side="right", padx=(8, 0))

        self._status_sep_right(si)

        self.status_time = ctk.CTkLabel(
            si, text="Image time: 0:00", font=(self.font_family, 11),
            text_color=FG_COLOR)
        self.status_time.pack(side="right", padx=(8, 0))

        self._status_sep_right(si)

        self.status_zoom = ctk.CTkLabel(
            si, text="Zoom: 100%", font=(self.font_family, 11),
            text_color=FG_COLOR)
        self.status_zoom.pack(side="right", padx=(8, 0))

    def _status_sep(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=20,
                           fg_color=BORDER_COLOR)
        sep.pack(side="left", padx=6, fill="y")

    def _status_sep_right(self, parent):
        sep = ctk.CTkFrame(parent, width=1, height=20,
                           fg_color=BORDER_COLOR)
        sep.pack(side="right", padx=6, fill="y")

    def _update_status(self):
        pct = int(self.scale * 100)
        self.status_zoom.configure(text=f"Zoom: {pct}%")

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
        c.bind("<ButtonPress-3>", self.on_right_click)
        c.bind("<Motion>", self._on_motion)

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
        r.bind("<Right>", lambda e: self.next_image())
        r.bind("<Left>", lambda e: self.prev_image())
        r.bind("<Escape>", self._on_escape)
        r.bind("<Control-z>", self.undo_last)
        r.bind("<Control-y>", self.redo_last)
        if sys.platform == "darwin":
            r.bind("<Command-z>", self.undo_last)
            r.bind("<Command-y>", self.redo_last)

        r.bind("h", lambda e: self._key_action(self.toggle_help))
        r.bind("m", lambda e: self._key_action(self._toggle_mode))
        r.bind("s", lambda e: self._key_action(self._toggle_snap))
        r.bind("v", lambda e: self._key_action(self._toggle_stream))

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

    # ──────────────────────────────────────────────────────────────────────────
    #  Mode toggle
    # ──────────────────────────────────────────────────────────────────────────
    def _toggle_mode(self, event=None):
        if self.mode == "box":
            self.mode = "polygon"
            self.mode_btn.configure(text="Mode: Polygon \u2b21")
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

        classes_file = os.path.join(folder, "classes.txt")
        if os.path.exists(classes_file):
            self.class_names = {}
            with open(classes_file, "r") as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        self.class_names[i] = name
            self._refresh_class_dropdown()

        self._load_class_colors()
        self._load_stats()
        self._load_completed_from_stats()
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

    def _has_annotations(self, img_name):
        stem = os.path.splitext(img_name)[0]
        for label_dir in (self.detect_dir, self.segment_dir):
            if label_dir is None:
                continue
            label_path = os.path.join(label_dir, f"{stem}.txt")
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, "r") as f:
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
            except Exception:
                pass

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

    # ──────────────────────────────────────────────────────────────────────────
    #  Quit
    # ──────────────────────────────────────────────────────────────────────────
    def _quit(self):
        if self.image_folder and self.images:
            print("[YoloLabeler] Saving and closing...")
            try:
                self._record_image_time()
                self.save_annotations()
                self._end_session()
                self._save_stats()
                print("[YoloLabeler] Writing CSV...")
                self.save_consolidated_csv()
                print("[YoloLabeler] Done.")
            except Exception as e:
                print(f"Warning: Could not save on exit: {e}")
        if self._timer_after_id:
            self.root.after_cancel(self._timer_after_id)
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

    # ──────────────────────────────────────────────────────────────────────────
    #  Time tracking
    # ──────────────────────────────────────────────────────────────────────────
    def _stats_path(self):
        if self.image_folder:
            return os.path.join(self.image_folder, "annotation_stats.json")
        return None

    def _load_stats(self):
        path = self._stats_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
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
            with open(path, "w") as f:
                json.dump(self._stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save stats: {e}")

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

        self._stats["sessions"].append({
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
        self._save_class_colors()
        return color

    def _count_class_annotations(self):
        """Count annotations per class across boxes and polygons."""
        counts = {}
        for *_, cls in self.boxes:
            counts[cls] = counts.get(cls, 0) + 1
        for _, cls in self.polygons:
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def _refresh_class_dropdown(self):
        counts = self._count_class_annotations()
        items = []
        for cid, name in sorted(self.class_names.items()):
            c = counts.get(cid, 0)
            items.append(f"{cid}: {name} ({c})")
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
        try:
            class_id = int(choice.split(":")[0].strip())
            self.active_class = class_id
            self._update_color_btn()
            self._refresh_class_dropdown()
            self.update_title()
        except (ValueError, IndexError):
            pass

    def _on_class_enter(self, event=None):
        text = self.class_dropdown.get().strip()
        if not text:
            return
        # Strip off the " (N)" count suffix if present (from dropdown display)
        text = re.sub(r'\s*\(\d+\)\s*$', '', text).strip()

        if ":" in text:
            try:
                class_id = int(text.split(":")[0].strip())
                if class_id in self.class_names:
                    self.active_class = class_id
                    self._refresh_class_dropdown()
                    self._update_color_btn()
                    self.update_title()
                    return
            except (ValueError, IndexError):
                pass

        for cid, cname in self.class_names.items():
            if cname.lower() == text.lower():
                self.active_class = cid
                self._refresh_class_dropdown()
                self._update_color_btn()
                self.update_title()
                return

        next_id = (max(self.class_names.keys()) + 1
                   if self.class_names else 0)
        self.class_names[next_id] = text
        self.active_class = next_id
        print(f"[YoloLabeler] New class added: {next_id}: {text}")
        self._refresh_class_dropdown()
        self._update_color_btn()
        self._save_classes_file()
        self.update_title()

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

    def _save_classes_file(self):
        if self.image_folder is None:
            return
        classes_file = os.path.join(self.image_folder, "classes.txt")
        try:
            with open(classes_file, "w") as f:
                for cid in sorted(self.class_names.keys()):
                    f.write(f"{self.class_names[cid]}\n")
        except OSError as e:
            print(f"Warning: Could not save classes file: {e}")

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
            if len(val) == 7 and val.startswith("#"):
                try:
                    int(val[1:], 16)
                    _update_preview(val)
                except ValueError:
                    pass

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
            self._save_class_colors()
            self.display_image()

    def _load_class_colors(self):
        if not self.image_folder:
            return
        path = os.path.join(self.image_folder, "class_colors.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self.class_colors = {int(k): v for k, v in data.items()}
            except Exception:
                self.class_colors = {}

    def _save_class_colors(self):
        if not self.image_folder:
            return
        path = os.path.join(self.image_folder, "class_colors.json")
        try:
            with open(path, "w") as f:
                json.dump({str(k): v for k, v in self.class_colors.items()},
                          f, indent=2)
        except OSError as e:
            print(f"Warning: Could not save class colors: {e}")

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
        if not self.images or self.image_folder is None:
            return
        if self.index >= len(self.images):
            self.index = 0
        if self.index < 0:
            self.index = len(self.images) - 1
        print(f"[YoloLabeler] Loading image {self.index + 1}/{len(self.images)}: {self.images[self.index]}")

        self.boxes = []
        self.polygons = []
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

        try:
            self.original_image = Image.open(img_path)
            self.original_image.load()
            self.original_image = auto_orient_image(self.original_image)
        except Exception as e:
            messagebox.showwarning("Image Error",
                                    f"Could not load:\n{img_path}\n\n{e}")
            self.index += 1
            self.load_image()
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

    def _fit_to_width(self, event=None):
        if not self.original_image:
            return
        if self.img_width <= 0 or self.img_height <= 0:
            return
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 750
        fit_scale = cw / self.img_width
        self.zoom_index = self._nearest_zoom_index(fit_scale)
        self.scale = self.zoom_levels[self.zoom_index]
        self.offset_x = (cw - self.img_width * self.scale) / 2
        self.offset_y = (ch - self.img_height * self.scale) / 2
        self._request_redraw()
        self._update_status()

    def _fit_to_height(self, event=None):
        if not self.original_image:
            return
        if self.img_width <= 0 or self.img_height <= 0:
            return
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 750
        fit_scale = ch / self.img_height
        self.zoom_index = self._nearest_zoom_index(fit_scale)
        self.scale = self.zoom_levels[self.zoom_index]
        self.offset_x = (cw - self.img_width * self.scale) / 2
        self.offset_y = (ch - self.img_height * self.scale) / 2
        self._request_redraw()
        self._update_status()

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
        if self.detect_dir is None or self.segment_dir is None:
            return
        stem = os.path.splitext(self.images[self.index])[0]

        detect_path = os.path.join(self.detect_dir, f"{stem}.txt")
        if os.path.exists(detect_path):
            try:
                with open(detect_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id = int(parts[0])
                        vals = [float(v) for v in parts[1:]]
                        xc = vals[0] * self.img_width
                        yc = vals[1] * self.img_height
                        w = vals[2] * self.img_width
                        h = vals[3] * self.img_height
                        self.boxes.append((
                            xc - w / 2, yc - h / 2,
                            xc + w / 2, yc + h / 2, class_id))
                        if class_id not in self.class_names:
                            self.class_names[class_id] = f"class_{class_id}"
                            self._refresh_class_dropdown()
            except Exception as e:
                print(f"Warning: Could not load detect labels for {stem}: {e}")

        segment_path = os.path.join(self.segment_dir, f"{stem}.txt")
        if os.path.exists(segment_path):
            try:
                with open(segment_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7 or len(parts) % 2 != 1:
                            continue
                        class_id = int(parts[0])
                        vals = [float(v) for v in parts[1:]]
                        points = []
                        for i in range(0, len(vals), 2):
                            px = vals[i] * self.img_width
                            py = vals[i + 1] * self.img_height
                            points.append((px, py))
                        self.polygons.append((points, class_id))
                        if class_id not in self.class_names:
                            self.class_names[class_id] = f"class_{class_id}"
                            self._refresh_class_dropdown()
            except Exception as e:
                print(f"Warning: Could not load segment labels for {stem}: {e}")

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
                self.displayed_image = cropped.resize(
                    (out_w, out_h),
                    Image.Resampling.BILINEAR if self._fast_resample
                    else Image.Resampling.LANCZOS)
                self._cached_tk_image = ImageTk.PhotoImage(
                    self.displayed_image)
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
        label_size = max(7, min(int(9 * (0.6 + s * 0.4)), 18))
        dash_a = max(2, int(4 * (0.5 + s * 0.5)))
        dash_b = max(2, int(4 * (0.5 + s * 0.5)))

        for box in self.boxes:
            x1, y1, x2, y2, class_id = box
            cx1, cy1 = self.image_to_canvas(x1, y1)
            cx2, cy2 = self.image_to_canvas(x2, y2)
            color = self._get_class_color(class_id)
            self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2, outline=color, width=line_w)
            class_name = self.class_names.get(class_id, str(class_id))
            self.canvas.create_text(
                cx1 + 2, cy1 - 2, anchor="sw",
                text=f"{class_id}: {class_name}",
                fill=color, font=(self.font_family, label_size, "bold"))

        for poly_idx, (points, class_id) in enumerate(self.polygons):
            color = self._get_class_color(class_id)
            is_selected = (poly_idx == self._selected_polygon_idx)
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
            if self.mode == "polygon":
                show_verts = (
                    is_selected
                    or poly_idx == self._hovered_polygon_idx
                    or (self._dragging_vertex is not None
                        and self._dragging_vertex[0] == poly_idx)
                )
                if show_verts:
                    for px, py in points:
                        cx, cy = self.image_to_canvas(px, py)
                        self.canvas.create_oval(
                            cx - vert_r, cy - vert_r,
                            cx + vert_r, cy + vert_r,
                            fill=draw_color, outline="white", width=1)
            if points:
                lx, ly = self.image_to_canvas(*points[0])
                class_name = self.class_names.get(class_id, str(class_id))
                self.canvas.create_text(
                    lx + 2, ly - 2, anchor="sw",
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
                "  m              Toggle Box/Polygon mode",
                "  0-9            Select class by ID",
                "  Ctrl+Z         Undo",
                "  Ctrl+Y         Redo",
                "  h              Toggle this help",
                "  \u2190/\u2192            Previous / Next image",
                "",
                "\u2500\u2500 Mouse \u2500\u2500",
                "  Left-click + drag    Draw bounding box",
                "  Right-click          Delete annotation",
                "  Ctrl+Scroll          Zoom at cursor",
                "  Scroll               Pan up/down",
                "  Shift+Scroll         Pan left/right",
                "  Middle-click         Pan (drag)",
            ]
        else:
            help_lines = [
                "\u2500\u2500 Keyboard \u2500\u2500",
                "  m              Toggle Box/Polygon mode",
                "  v              Toggle vertex streaming",
                "  s              Toggle vertex snapping",
                "  0-9            Select class by ID",
                "  Ctrl+Z         Undo",
                "  Ctrl+Y         Redo",
                "  h              Toggle this help",
                "  Escape         Cancel / Deselect polygon",
                "  \u2190/\u2192            Previous / Next image",
                "",
                "\u2500\u2500 Mouse \u2500\u2500",
                "  Left-click           Place vertex / Select polygon",
                "  Double-click         Close polygon",
                "  Right-click          Delete annotation / vertex",
                "  Drag vertex          Move vertex (selected polygon)",
                "  Click edge           Insert vertex (selected polygon)",
                "  Ctrl+Scroll          Zoom at cursor",
                "  Scroll               Pan up/down",
                "  Shift+Scroll         Pan left/right",
                "  Middle-click         Pan (drag)",
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

        # Update snap indicator
        if self.mode == "polygon" and self.snap_enabled:
            ix, iy = self.canvas_to_image(event.x, event.y)
            snapped = self._maybe_snap(ix, iy)
            if snapped != (ix, iy):
                sx, sy = self.image_to_canvas(*snapped)
                snap_r = max(4, min(SNAP_RADIUS * (1.6 - self.scale * 0.2), 14))
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
        else:
            if self._snap_indicator_item:
                try:
                    self.canvas.delete(self._snap_indicator_item)
                except tk.TclError:
                    pass
                self._snap_indicator_item = None

        # Polygon hover detection (for showing vertices on adjacent polygons)
        if self.mode == "polygon":
            ix, iy = self.canvas_to_image(event.x, event.y)
            new_hover = None
            hover_thr = 25  # canvas-pixel proximity to reveal vertices
            if not self.current_polygon:
                # Not drawing — full hit-test (vertex, edge, proximity, point-in-polygon)
                vhit = self._find_nearest_vertex(event.x, event.y, threshold=hover_thr)
                if vhit:
                    new_hover = vhit[0]
                else:
                    ehit = self._find_nearest_edge(event.x, event.y, threshold=hover_thr)
                    if ehit:
                        new_hover = ehit[0]
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
                    ehit = self._find_nearest_edge(event.x, event.y, threshold=hover_thr)
                    if ehit:
                        new_hover = ehit[0]
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
    def _maybe_snap(self, ix, iy):
        if not self.snap_enabled:
            return (ix, iy)
        cx, cy = self.image_to_canvas(ix, iy)
        best_dist = SNAP_RADIUS
        best_pt = None
        for points, _ in self.polygons:
            for px, py in points:
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
        cx, cy = self.image_to_canvas(ix, iy)
        best_dist = SNAP_RADIUS
        best_pt = None
        for points, _ in self.polygons:
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
                    self._selected_polygon_idx = None
                    self._clear_drag_state()
                    self._mark_image_annotated()
                    self.display_image()
                    self.update_title()
                    return
            # Click outside selected polygon = deselect
            self._selected_polygon_idx = None
            self.display_image()
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
                # Vertex drag on the selected polygon
                vertex_hit = self._find_nearest_vertex(event.x, event.y)
                if vertex_hit and vertex_hit[0] == pi:
                    vi = vertex_hit[1]
                    self._push_undo()
                    self._dragging_vertex = (pi, vi)
                    self._drag_orig_pos = self.polygons[pi][0][vi]
                    self.canvas.config(cursor="fleur")
                    return
                # Edge insert on the selected polygon
                edge_hit = self._find_nearest_edge(event.x, event.y)
                if edge_hit and edge_hit[0] == pi:
                    ei, insert_pt = edge_hit[1], edge_hit[2]
                    self._push_undo()
                    points, cls = self.polygons[pi]
                    new_points = list(points)
                    new_points.insert(ei + 1, insert_pt)
                    self.polygons[pi] = (new_points, cls)
                    self._dragging_vertex = (pi, ei + 1)
                    self._drag_orig_pos = insert_pt
                    self.canvas.config(cursor="fleur")
                    self.display_image()
                    return
            # Click not on selected polygon's vertex/edge — deselect
            self._selected_polygon_idx = None
            self.display_image()
            return

        # ── Not drawing, nothing selected — check if clicking on polygon ──
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
            ix, iy = self.canvas_to_image(event.x, event.y)
            ix, iy = self._maybe_snap(ix, iy)
            ix = max(0, min(self.img_width, ix))
            iy = max(0, min(self.img_height, iy))
            points, cls = self.polygons[pi]
            new_points = list(points)
            new_points[vi] = (ix, iy)
            self.polygons[pi] = (new_points, cls)
            self.display_image()

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
        self._mark_image_annotated()
        self._record_annotation_added()
        self.current_polygon = []
        self._poly_preview_line = None
        self.display_image()
        self.update_title()

    # ──────────────────────────────────────────────────────────────────────────
    #  Polygon geometry helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _find_nearest_vertex(self, cx, cy, threshold=8):
        best = None
        best_dist = threshold
        for pi, (points, _) in enumerate(self.polygons):
            for vi, (px, py) in enumerate(points):
                vcx, vcy = self.image_to_canvas(px, py)
                dist = math.hypot(cx - vcx, cy - vcy)
                if dist < best_dist:
                    best_dist = dist
                    best = (pi, vi)
        return best

    def _find_nearest_edge(self, cx, cy, threshold=6):
        best = None
        best_dist = threshold
        for pi, (points, _) in enumerate(self.polygons):
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

    @staticmethod
    def _point_in_polygon(px, py, points):
        n = len(points)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = points[i]
            xj, yj = points[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

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
            copy.deepcopy(self.polygons),
            self._selected_polygon_idx,
        )
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def undo_last(self, event=None):
        # While drawing, undo one vertex at a time
        if self.current_polygon:
            pt = self.current_polygon.pop()
            self._vertex_redo_stack.append(pt)
            self.display_image()
            return
        # Otherwise restore from snapshot stack
        if not self._undo_stack:
            return
        # Push current state to redo
        redo_snapshot = (
            list(self.boxes),
            copy.deepcopy(self.polygons),
            self._selected_polygon_idx,
        )
        self._redo_stack.append(redo_snapshot)
        # Restore previous state
        boxes, polygons, sel_idx = self._undo_stack.pop()
        self.boxes = boxes
        self.polygons = polygons
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
            copy.deepcopy(self.polygons),
            self._selected_polygon_idx,
        )
        self._undo_stack.append(undo_snapshot)
        boxes, polygons, sel_idx = self._redo_stack.pop()
        self.boxes = boxes
        self.polygons = polygons
        self._selected_polygon_idx = sel_idx
        self._clear_drag_state()
        self.display_image()
        self.update_title()

    # ──────────────────────────────────────────────────────────────────────────
    #  Save annotations
    # ──────────────────────────────────────────────────────────────────────────
    def save_annotations(self):
        if not self.images or self.labels_dir is None or self.detect_dir is None or self.segment_dir is None:
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
            if self.boxes:
                with open(detect_path, "w") as f:
                    for x1, y1, x2, y2, cls in self.boxes:
                        xc = ((x1 + x2) / 2) / img_w
                        yc = ((y1 + y2) / 2) / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h
                        f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            elif os.path.exists(detect_path):
                os.remove(detect_path)

            if self.polygons:
                with open(segment_path, "w") as f:
                    for points, cls in self.polygons:
                        coords = " ".join(
                            f"{x / img_w:.6f} {y / img_h:.6f}"
                            for x, y in points)
                        f.write(f"{cls} {coords}\n")
            elif os.path.exists(segment_path):
                os.remove(segment_path)
        except OSError as e:
            print(f"Warning: Could not save annotations: {e}")

    def _get_oriented_size(self, img_path):
        """Get image dimensions accounting for EXIF orientation without loading pixels."""
        try:
            img = Image.open(img_path)
            w, h = img.size
            try:
                exif = getattr(img, '_getexif', lambda: None)()
                if exif:
                    for k, v in ExifTags.TAGS.items():
                        if v == "Orientation":
                            orient = exif.get(k, 1)
                            if orient in (5, 6, 7, 8):
                                w, h = h, w
                            break
            except Exception:
                pass
            img.close()
            return w, h
        except Exception:
            return None

    def save_consolidated_csv(self):
        if not self.image_folder or not self.labels_dir or self.detect_dir is None or self.segment_dir is None:
            return
        csv_path = os.path.join(self.image_folder, "annotations.csv")
        count = 0
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image", "type", "class_id", "class_name",
                "x1", "y1", "x2", "y2", "polygon_points"])
            for img_name in self.images:
                stem = os.path.splitext(img_name)[0]

                # Use cached dims or read header only (no pixel loading)
                if img_name in self._image_dims:
                    iw, ih = self._image_dims[img_name]
                else:
                    dims = self._get_oriented_size(
                        os.path.join(self.image_folder, img_name))
                    if dims is None:
                        continue
                    iw, ih = dims
                    self._image_dims[img_name] = (iw, ih)

                detect_path = os.path.join(
                    self.detect_dir, f"{stem}.txt")
                if os.path.exists(detect_path):
                    with open(detect_path, "r") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            cls = int(parts[0])
                            cn = self.class_names.get(
                                cls, f"class_{cls}")
                            vals = [float(v) for v in parts[1:]]
                            xc, yc = vals[0] * iw, vals[1] * ih
                            bw, bh = vals[2] * iw, vals[3] * ih
                            writer.writerow([
                                img_name, "box", cls, cn,
                                f"{xc - bw/2:.1f}",
                                f"{yc - bh/2:.1f}",
                                f"{xc + bw/2:.1f}",
                                f"{yc + bh/2:.1f}", ""])
                            count += 1

                segment_path = os.path.join(
                    self.segment_dir, f"{stem}.txt")
                if os.path.exists(segment_path):
                    with open(segment_path, "r") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) < 7 or len(parts) % 2 != 1:
                                continue
                            cls = int(parts[0])
                            cn = self.class_names.get(
                                cls, f"class_{cls}")
                            vals = [float(v) for v in parts[1:]]
                            pts, xs, ys = [], [], []
                            for i in range(0, len(vals), 2):
                                px = vals[i] * iw
                                py = vals[i + 1] * ih
                                pts.append(
                                    f"({px:.1f}, {py:.1f})")
                                xs.append(px)
                                ys.append(py)
                            pts_str = "[" + ", ".join(pts) + "]"
                            writer.writerow([
                                img_name, "polygon", cls, cn,
                                f"{min(xs):.1f}",
                                f"{min(ys):.1f}",
                                f"{max(xs):.1f}",
                                f"{max(ys):.1f}", pts_str])
                            count += 1


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
