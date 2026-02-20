"""
BoxLabeler - Image Annotation Tool for YOLO Training

Features:
- Draw bounding boxes on images with class labels
- Save annotations in both YOLO .txt format (normalized) and CSV
- Multi-class support via dropdown + text entry (define new classes on the fly)
- Ctrl+Scroll to zoom in/out at cursor position
- Scroll to pan up/down, Shift+Scroll to pan left/right
- Middle-click drag to pan
- Undo last box with 'z'
- Right-click a box to delete it
- On-screen help overlay with dark background (toggle with 'h')
- Auto EXIF orientation correction
- Boundary clamping for boxes
- Error handling for corrupt/unreadable images
- Auto-resume at first unlabeled image
- Verbose debugging to stdout
"""

import os
import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ExifTags


# Default class names (editable). Index = class_id.
DEFAULT_CLASS_NAMES = {0: "object"}

# Colors for different classes (cycles if more classes than colors)
CLASS_COLORS = [
    "red", "blue", "green", "orange", "purple",
    "cyan", "magenta", "yellow", "lime", "pink",
]


def auto_orient_image(img):
    """Apply EXIF orientation tag so images display correctly."""
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
        if orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 7:
            img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img


class BoxLabeler:
    def __init__(self, root, image_folder=None, class_names=None):
        self.root = root
        self.image_folder = image_folder
        self.class_names = dict(class_names) if class_names else dict(DEFAULT_CLASS_NAMES)
        self.images = []
        self.labels_dir = None
        self.img_width = 0
        self.img_height = 0
        self.original_image = None

        # State variables
        self.boxes = []
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.active_class = 0
        self.show_help = True

        # View transform
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # Cached rendered image (avoid re-rendering on pan)
        self._cached_scale = None
        self._cached_tk_image = None

        # Predefined zoom levels (snaps to these on ctrl+scroll)
        self.zoom_levels = [
            0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67,
            0.75, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0
        ]
        self.zoom_index = 0  # Set when image loads via _fit_to_window

        # Middle-click pan state
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_start_offset_x = None
        self.pan_start_offset_y = None

        # Throttle flag for redraw
        self._redraw_pending = False

        # Image index
        self.index = 0

        # ---- Top toolbar frame ----
        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side="top", fill="x", padx=4, pady=4)

        tk.Label(self.toolbar, text="Class:").pack(side="left", padx=(0, 4))

        self.class_var = tk.StringVar()
        self.class_dropdown = ttk.Combobox(
            self.toolbar, textvariable=self.class_var, state="readonly", width=20
        )
        self.class_dropdown.pack(side="left", padx=(0, 8))
        self.class_dropdown.bind("<<ComboboxSelected>>", self._on_class_selected)
        self._refresh_class_dropdown()

        tk.Label(self.toolbar, text="New class:").pack(side="left", padx=(8, 4))
        self.new_class_entry = tk.Entry(self.toolbar, width=18)
        self.new_class_entry.pack(side="left", padx=(0, 4))
        tk.Button(self.toolbar, text="Add", command=self._add_new_class).pack(
            side="left", padx=(0, 8)
        )
        self.new_class_entry.bind("<Return>", lambda e: self._add_new_class())

        tk.Button(self.toolbar, text="\U0001f4c2 Open Folder", command=self._open_folder).pack(
            side="left", padx=(8, 0)
        )

        self.counter_label = tk.Label(self.toolbar, text="")
        self.counter_label.pack(side="right", padx=(8, 0))

        tk.Button(self.toolbar, text="Next \u25b6", command=self.next_image).pack(
            side="right", padx=2
        )
        tk.Button(self.toolbar, text="\u25c0 Prev", command=self.prev_image).pack(
            side="right", padx=2
        )

        # Canvas
        self.canvas = tk.Canvas(root, cursor="cross", bg="#333333")
        self.canvas.pack(fill="both", expand=True)

        # ---- Bindings ----
        # Mouse
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)

        # Middle-click pan
        self.canvas.bind("<ButtonPress-2>", self.on_middle_press)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_release)

        # Scroll (Windows/macOS)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

        # Scroll (Linux)
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)
        self.canvas.bind("<Control-Button-4>", self._on_ctrl_mousewheel_linux)
        self.canvas.bind("<Control-Button-5>", self._on_ctrl_mousewheel_linux)
        self.canvas.bind("<Shift-Button-4>", self._on_shift_mousewheel_linux)
        self.canvas.bind("<Shift-Button-5>", self._on_shift_mousewheel_linux)

        # Keyboard
        root.bind("<Right>", lambda e: self.next_image())
        root.bind("<Left>", lambda e: self.prev_image())
        root.bind("<Escape>", lambda e: self._quit())
        root.bind("z", self.undo_box)
        root.bind("h", self.toggle_help)

        # Also catch the window X button
        root.protocol("WM_DELETE_WINDOW", self._quit)

        # Focus canvas for scroll events
        self.canvas.focus_set()
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())

        # Deferred first load so canvas has real dimensions
        if self.image_folder:
            self._init_folder(self.image_folder)
            self.root.after(100, self.load_image)
        else:
            # Show welcome message on empty canvas
            self.root.after(100, self._show_welcome)

    # --------------------------------------------------
    # Folder initialization
    # --------------------------------------------------
    def _init_folder(self, folder):
        """Initialize image list and labels dir for a folder."""
        self.image_folder = folder
        self.images = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ])
        self.labels_dir = os.path.join(folder, "labels")
        os.makedirs(self.labels_dir, exist_ok=True)
        print(f"[DEBUG] Found {len(self.images)} images in {folder}")
        print(f"[DEBUG] Labels directory: {self.labels_dir}")

        # Load classes.txt if present
        classes_file = os.path.join(folder, "classes.txt")
        if os.path.exists(classes_file):
            self.class_names = {}
            with open(classes_file, "r") as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        self.class_names[i] = name
            self._refresh_class_dropdown()
            print(f"[DEBUG] Loaded {len(self.class_names)} class names from {classes_file}")

        self.index = self._find_resume_index()

    def _find_resume_index(self):
        """Find the index of the first image without a label file."""
        for i, img_name in enumerate(self.images):
            stem = os.path.splitext(img_name)[0]
            label_path = os.path.join(self.labels_dir, f"{stem}.txt")
            if not os.path.exists(label_path):
                print(f"[DEBUG] Resuming at image {i + 1}/{len(self.images)}: {img_name}")
                return i
        print(f"[DEBUG] All {len(self.images)} images have labels. Starting at last image.")
        return len(self.images) - 1

    def _show_welcome(self):
        """Show a welcome message when no folder is loaded."""
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 800
        self.canvas.create_text(
            cw // 2, ch // 2,
            text="Click \"\U0001f4c2 Open Folder\" to load images",
            fill="white", font=("Arial", 16),
        )
        self.root.title("BoxLabeler")

    # --------------------------------------------------
    # Open folder
    # --------------------------------------------------
    def _open_folder(self):
        """Open a new image folder mid-session."""
        new_folder = filedialog.askdirectory(title="Select Folder of Images")
        if not new_folder:
            return

        # Save current work first
        if self.images:
            try:
                self.save_boxes()
            except Exception:
                pass

        self._init_folder(new_folder)

        if not self.images:
            messagebox.showinfo("No images", "No images found in the folder!")
            return

        print(f"[DEBUG] Opened new folder: {new_folder} ({len(self.images)} images)")
        self.load_image()

    # --------------------------------------------------
    # Quit
    # --------------------------------------------------
    def _quit(self):
        """Save current work and exit."""
        try:
            self.save_boxes()
            print("[DEBUG] Saved boxes before exit.")
        except Exception as e:
            print(f"[WARNING] Could not save on exit: {e}")
        self.root.destroy()

    # --------------------------------------------------
    # Class management
    # --------------------------------------------------
    def _refresh_class_dropdown(self):
        """Rebuild dropdown values from self.class_names."""
        items = [f"{cid}: {name}" for cid, name in sorted(self.class_names.items())]
        self.class_dropdown["values"] = items
        active_label = f"{self.active_class}: {self.class_names.get(self.active_class, '?')}"
        if active_label in items:
            self.class_dropdown.set(active_label)
        elif items:
            self.class_dropdown.current(0)

    def _on_class_selected(self, event=None):
        """Handle dropdown selection."""
        sel = self.class_var.get()
        try:
            class_id = int(sel.split(":")[0].strip())
            self.active_class = class_id
            print(f"[DEBUG] Selected class {class_id} ({self.class_names.get(class_id, '?')})")
            self.update_title()
        except (ValueError, IndexError):
            pass

    def _add_new_class(self):
        """Add a new class from the text entry."""
        name = self.new_class_entry.get().strip()
        if not name:
            return
        # Check if name already exists
        for cid, cname in self.class_names.items():
            if cname.lower() == name.lower():
                self.active_class = cid
                self._refresh_class_dropdown()
                self.new_class_entry.delete(0, tk.END)
                self.update_title()
                print(f"[DEBUG] Class '{name}' already exists as id {cid}")
                return
        next_id = max(self.class_names.keys()) + 1 if self.class_names else 0
        self.class_names[next_id] = name
        self.active_class = next_id
        self.new_class_entry.delete(0, tk.END)
        self._refresh_class_dropdown()
        self._save_classes_file()
        self.update_title()
        print(f"[DEBUG] Added new class {next_id}: {name}")

    def _save_classes_file(self):
        """Persist class names to classes.txt so they survive restarts."""
        classes_file = os.path.join(self.image_folder, "classes.txt")
        with open(classes_file, "w") as f:
            for cid in sorted(self.class_names.keys()):
                f.write(f"{self.class_names[cid]}\n")
        print(f"[DEBUG] Saved classes.txt ({len(self.class_names)} classes)")

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    def update_title(self):
        """Update window title and counter label."""
        class_name = self.class_names.get(self.active_class, f"class_{self.active_class}")
        labeled_count = sum(
            1
            for img in self.images
            if os.path.exists(
                os.path.join(self.labels_dir, f"{os.path.splitext(img)[0]}.txt")
            )
        )
        self.root.title(
            f"BoxLabeler \u2014 {self.images[self.index]} | "
            f"Class: {self.active_class} ({class_name}) | "
            f"Boxes: {len(self.boxes)}"
        )
        self.counter_label.config(
            text=f"Image {self.index + 1} / {len(self.images)}  |  Labeled: {labeled_count}"
        )

    # --------------------------------------------------
    # Coordinate conversions
    # --------------------------------------------------
    def canvas_to_image(self, cx, cy):
        """Convert canvas pixel coords to image pixel coords."""
        ix = (cx - self.offset_x) / self.scale
        iy = (cy - self.offset_y) / self.scale
        return ix, iy

    def image_to_canvas(self, ix, iy):
        """Convert image pixel coords to canvas pixel coords."""
        cx = ix * self.scale + self.offset_x
        cy = iy * self.scale + self.offset_y
        return cx, cy

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    def load_image(self):
        """Load the current image by self.index."""
        if not self.images:
            return
        if self.index >= len(self.images):
            messagebox.showinfo("Done", "All images labeled!")
            print("[DEBUG] All images labeled. Exiting.")
            self.root.destroy()
            return

        # Reset state for new image
        self.boxes = []
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._cached_scale = None
        self._cached_tk_image = None

        img_path = os.path.join(self.image_folder, self.images[self.index])
        print(f"[DEBUG] Loading image {self.index + 1}/{len(self.images)}: {img_path}")

        try:
            self.original_image = Image.open(img_path)
            self.original_image.load()  # Force load to catch corrupt files
            self.original_image = auto_orient_image(self.original_image)
        except Exception as e:
            print(f"[ERROR] Failed to load image {img_path}: {e}")
            messagebox.showwarning("Image Error", f"Could not load:\n{img_path}\n\n{e}")
            self.index += 1
            self.load_image()
            return

        self.img_width, self.img_height = self.original_image.size
        self._fit_to_window()
        self._load_existing_labels()
        self.display_image()
        self.update_title()

    def _fit_to_window(self):
        """Set scale and offset so the image fits the canvas/window."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        # Fallback if canvas hasn't rendered yet
        if cw < 10:
            cw = 1200
        if ch < 10:
            ch = 750
        sx = cw / self.img_width
        sy = ch / self.img_height
        fit_scale = min(sx, sy, 1.0)  # Don't upscale beyond 1:1
        # Snap to nearest predefined zoom level
        self.zoom_index = self._nearest_zoom_index(fit_scale)
        self.scale = self.zoom_levels[self.zoom_index]
        # Center the image
        self.offset_x = (cw - self.img_width * self.scale) / 2
        self.offset_y = (ch - self.img_height * self.scale) / 2

    def _nearest_zoom_index(self, target_scale):
        """Find the index of the zoom level closest to target_scale."""
        best_idx = 0
        best_diff = abs(self.zoom_levels[0] - target_scale)
        for i, level in enumerate(self.zoom_levels):
            diff = abs(level - target_scale)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    # --------------------------------------------------
    # Load existing YOLO labels
    # --------------------------------------------------
    def _load_existing_labels(self):
        """Load existing YOLO .txt labels back into boxes list."""
        label_path = os.path.join(
            self.labels_dir, f"{os.path.splitext(self.images[self.index])[0]}.txt"
        )
        if not os.path.exists(label_path):
            return

        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * self.img_width
                    y_center = float(parts[2]) * self.img_height
                    w = float(parts[3]) * self.img_width
                    h = float(parts[4]) * self.img_height
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    self.boxes.append((x1, y1, x2, y2, class_id))

                    if class_id not in self.class_names:
                        self.class_names[class_id] = f"class_{class_id}"
                        self._refresh_class_dropdown()

            print(f"[DEBUG] Loaded {len(self.boxes)} existing labels from {label_path}")
        except Exception as e:
            print(f"[WARNING] Could not load labels from {label_path}: {e}")

    # --------------------------------------------------
    # Display (throttled)
    # --------------------------------------------------
    def _request_redraw(self):
        """Coalesce rapid redraw requests into a single display_image call."""
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after_idle(self._do_redraw)

    def _do_redraw(self):
        """Execute the coalesced redraw."""
        self._redraw_pending = False
        self.display_image()

    def display_image(self):
        """Render only the visible viewport crop of the image (memory-safe)."""
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 800

        # Determine visible region in image coordinates
        vis_x1, vis_y1 = self.canvas_to_image(0, 0)
        vis_x2, vis_y2 = self.canvas_to_image(cw, ch)

        # Clamp to image bounds
        crop_x1 = max(0, int(vis_x1))
        crop_y1 = max(0, int(vis_y1))
        crop_x2 = min(self.img_width, int(vis_x2) + 1)
        crop_y2 = min(self.img_height, int(vis_y2) + 1)

        # Build a cache key from scale + crop region
        cache_key = (self.scale, crop_x1, crop_y1, crop_x2, crop_y2)

        if self._cached_scale != cache_key:
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            if crop_w > 0 and crop_h > 0:
                cropped = self.original_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                out_w = max(int(crop_w * self.scale), 1)
                out_h = max(int(crop_h * self.scale), 1)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                self.displayed_image = cropped.resize((out_w, out_h), resample)
                self._cached_tk_image = ImageTk.PhotoImage(self.displayed_image)
            else:
                self._cached_tk_image = None

            self._cached_scale = cache_key

        self.canvas.delete("all")

        if self._cached_tk_image is not None:
            # Place the crop at its correct canvas position
            place_x = self.offset_x + crop_x1 * self.scale
            place_y = self.offset_y + crop_y1 * self.scale
            self.canvas.create_image(
                place_x, place_y, anchor="nw", image=self._cached_tk_image
            )

        # Draw boxes
        for box in self.boxes:
            x1, y1, x2, y2, class_id = box
            cx1, cy1 = self.image_to_canvas(x1, y1)
            cx2, cy2 = self.image_to_canvas(x2, y2)
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2)
            class_name = self.class_names.get(class_id, str(class_id))
            self.canvas.create_text(
                cx1 + 2, cy1 - 2, anchor="sw",
                text=f"{class_id}: {class_name}",
                fill=color, font=("Arial", 9, "bold"),
            )

        # Help overlay
        self.draw_help_overlay()

    def draw_help_overlay(self):
        """Draw key bindings on canvas with dark background."""
        if not self.show_help:
            return
        class_name = self.class_names.get(self.active_class, f"class_{self.active_class}")
        help_lines = [
            f"Active class: {self.active_class} ({class_name})",
            "",
            "Controls:",
            "  Left-click + drag     : Draw box",
            "  Right-click on box    : Delete box",
            "  Right/Left arrow      : Next/Prev image (saves)",
            "  Scroll                : Pan up/down",
            "  Shift+Scroll          : Pan left/right",
            "  Ctrl+Scroll           : Zoom at cursor",
            "  Middle-click + drag   : Pan",
            "  z                     : Undo last box",
            "  h                     : Toggle this help",
            "  Escape                : Quit",
        ]

        font_family = "Consolas"
        font_size = 10
        line_height = 16
        pad = 10
        # Use a generous character width for Consolas 10pt
        char_width = font_size * 0.65

        block_w = int(max(len(line) for line in help_lines) * char_width + pad * 3)
        block_h = len(help_lines) * line_height + pad * 2

        x0, y0 = 10, 10

        self.canvas.create_rectangle(
            x0, y0, x0 + block_w, y0 + block_h,
            fill="black", outline="gray", width=1, stipple="gray50",
        )

        for i, line in enumerate(help_lines):
            self.canvas.create_text(
                x0 + pad, y0 + pad + i * line_height,
                anchor="nw", text=line,
                fill="white", font=(font_family, font_size),
            )

    def toggle_help(self, event=None):
        """Toggle the help overlay."""
        self.show_help = not self.show_help
        self.display_image()

    # --------------------------------------------------
    # Scroll / Zoom / Pan
    # --------------------------------------------------
    def _on_mousewheel(self, event):
        """Scroll up/down -> pan vertically."""
        delta = event.delta
        if sys.platform == "darwin":
            self.offset_y += delta * 2
        else:
            self.offset_y += (delta // 120) * 40
        self._request_redraw()

    def _on_shift_mousewheel(self, event):
        """Shift+Scroll -> pan horizontally."""
        delta = event.delta
        if sys.platform == "darwin":
            self.offset_x += delta * 2
        else:
            self.offset_x += (delta // 120) * 40
        self._request_redraw()

    def _on_ctrl_mousewheel(self, event):
        """Ctrl+Scroll -> zoom at cursor position (stepped)."""
        direction = 1 if event.delta > 0 else -1
        self._zoom_step(event.x, event.y, direction)

    # Linux scroll equivalents
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
        """Step one zoom level up or down, keeping the cursor point fixed."""
        ix, iy = self.canvas_to_image(cx, cy)
        new_index = self.zoom_index + direction
        new_index = max(0, min(new_index, len(self.zoom_levels) - 1))
        if new_index == self.zoom_index:
            return  # Already at min/max, skip redraw
        self.zoom_index = new_index
        self.scale = self.zoom_levels[self.zoom_index]
        self.offset_x = cx - ix * self.scale
        self.offset_y = cy - iy * self.scale
        self._request_redraw()

    # Middle-click pan
    def on_middle_press(self, event):
        """Start middle-click pan."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.pan_start_offset_x = self.offset_x
        self.pan_start_offset_y = self.offset_y
        self.canvas.config(cursor="fleur")

    def on_middle_drag(self, event):
        """Handle middle-click drag for panning."""
        if self.pan_start_x is None:
            return
        self.offset_x = self.pan_start_offset_x + (event.x - self.pan_start_x)
        self.offset_y = self.pan_start_offset_y + (event.y - self.pan_start_y)
        self._request_redraw()

    def on_middle_release(self, event):
        """End middle-click pan."""
        self.pan_start_x = None
        self.pan_start_y = None
        self.canvas.config(cursor="cross")

    # --------------------------------------------------
    # Mouse events (box drawing)
    # --------------------------------------------------
    def on_button_press(self, event):
        """Start drawing a box."""
        self.start_x = event.x
        self.start_y = event.y
        color = CLASS_COLORS[self.active_class % len(CLASS_COLORS)]
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=color, width=2,
        )

    def on_move_press(self, event):
        """Update box preview while dragging."""
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        """Finish drawing a box."""
        if self.start_x is None or self.start_y is None:
            return

        # Convert canvas coords to image coords with clamping
        ix1, iy1 = self.canvas_to_image(self.start_x, self.start_y)
        ix2, iy2 = self.canvas_to_image(event.x, event.y)

        x1 = max(0, min(ix1, ix2))
        y1 = max(0, min(iy1, iy2))
        x2 = min(self.img_width, max(ix1, ix2))
        y2 = min(self.img_height, max(iy1, iy2))

        # Ignore tiny accidental clicks (less than 3px in image space)
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            print("[DEBUG] Box too small, ignoring")
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = None
            return

        self.boxes.append((x1, y1, x2, y2, self.active_class))
        print(
            f"[DEBUG] Added box: class={self.active_class} "
            f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )
        self.rect = None
        self.display_image()
        self.update_title()

    def on_right_click(self, event):
        """Delete the box under the right-click location."""
        if not self.boxes:
            return
        click_x, click_y = self.canvas_to_image(event.x, event.y)
        for i, (x1, y1, x2, y2, _) in enumerate(self.boxes):
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                removed = self.boxes.pop(i)
                print(f"[DEBUG] Right-click deleted box {i}: {removed}")
                self.display_image()
                self.update_title()
                return
        print(f"[DEBUG] Right-click at ({click_x:.1f}, {click_y:.1f}) \u2014 no box found")

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------
    def next_image(self, event=None):
        """Save current boxes and advance to next image."""
        self.save_boxes()
        self.index += 1
        print(f"[DEBUG] Moving to next image: index={self.index}")
        self.load_image()

    def prev_image(self, event=None):
        """Save current boxes and go back to previous image."""
        if self.index > 0:
            self.save_boxes()
            self.index -= 1
            print(f"[DEBUG] Moving to previous image: index={self.index}")
            self.load_image()

    # --------------------------------------------------
    # Undo box
    # --------------------------------------------------
    def undo_box(self, event=None):
        """Remove the last drawn box."""
        if self.boxes:
            removed = self.boxes.pop()
            print(f"[DEBUG] Undo last box: {removed}")
            self.display_image()
            self.update_title()

    # --------------------------------------------------
    # Save boxes (YOLO .txt + CSV)
    # --------------------------------------------------
    def save_boxes(self):
        """Save current boxes in YOLO .txt format and CSV."""
        if not self.images or self.labels_dir is None:
            return
        # Ensure labels directory exists (handles network paths)
        os.makedirs(self.labels_dir, exist_ok=True)

        stem = os.path.splitext(self.images[self.index])[0]
        img_w = self.img_width
        img_h = self.img_height

        # YOLO .txt format (class x_center y_center width height, normalized)
        yolo_path = os.path.join(self.labels_dir, f"{stem}.txt")
        with open(yolo_path, "w") as f:
            for x1, y1, x2, y2, cls in self.boxes:
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        print(f"[DEBUG] Saved {len(self.boxes)} boxes (YOLO) to {yolo_path}")

        # CSV format (absolute pixel coords)
        csv_path = os.path.join(self.labels_dir, f"{stem}_boxes.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "x1", "y1", "x2", "y2"])
            for x1, y1, x2, y2, cls in self.boxes:
                class_name = self.class_names.get(cls, f"class_{cls}")
                writer.writerow([cls, class_name, x1, y1, x2, y2])
        print(f"[DEBUG] Saved {len(self.boxes)} boxes (CSV) to {csv_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    """Entry point for the boxlabeler command."""
    root = tk.Tk()
    root.geometry("1200x800")
    root.withdraw()

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        if not os.path.isdir(folder):
            print(f"[ERROR] Invalid folder: {folder}")
            sys.exit(1)
    else:
        folder = filedialog.askdirectory(title="Select Folder of Images")
        if not folder:
            print("[ERROR] No folder selected. Exiting.")
            sys.exit(1)

    root.deiconify()
    print(f"[DEBUG] Starting BoxLabeler in folder: {folder}")
    app = BoxLabeler(root, image_folder=folder)  # noqa: F841
    root.mainloop()
    print("[DEBUG] BoxLabeler exited.")


if __name__ == "__main__":
    main()
