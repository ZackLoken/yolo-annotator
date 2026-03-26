# YoloLabeler

![YoloLabeler GUI](GUI.png)

Lightweight desktop tool for drawing YOLO bounding-box and instance segmentation
annotations on images.
Built with Python + CustomTkinter — no GPU, no server, no browser required.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Install

```bash
pip install git+https://github.com/ZackLoken/yolo-annotator.git
```

Or for development:

```bash
git clone https://github.com/ZackLoken/yolo-annotator.git
cd yolo-annotator
pip install -e .
```

### Requirements

- Python 3.9+
- Pillow ≥ 9.0
- CustomTkinter ≥ 5.0

---

## Usage

**Launch the GUI:**

```bash
yololabeler
```

A folder dialog will open — select a folder of images.

**Or pass a folder directly:**

```bash
yololabeler /path/to/images
```

**Or run as a module:**

```bash
python -m yololabeler /path/to/images
```

> The legacy `boxlabeler` command also works as an alias.

---

## Controls

| Action                       | Input                    |
|------------------------------|--------------------------|
| Toggle Box / Polygon mode    | `m`                      |
| Toggle vertex streaming      | `v`                      |
| Toggle vertex snapping       | `s`                      |
| Select class by id           | `0`–`9`                  |
| Undo                         | `Ctrl+Z`                 |
| Redo                         | `Ctrl+Y`                 |
| Toggle help overlay          | `h`                      |
| Cancel / Deselect polygon    | `Escape`                 |
| Next / Previous image        | `→` / `←`                |
| **Box mode**                 |                          |
| Draw a box                   | Left-click + drag        |
| Delete a box                 | Right-click on box       |
| **Polygon mode**             |                          |
| Place vertex                 | Left-click               |
| Select polygon               | Left-click on polygon    |
| Close polygon                | Double-click             |
| Move vertex                  | Drag vertex (selected)   |
| Insert vertex on edge        | Click edge (selected)    |
| Delete vertex                | Right-click vertex       |
| Delete polygon               | Right-click in polygon   |
| **Navigation & view**        |                          |
| Pan up / down                | Scroll                   |
| Pan left / right             | Shift + Scroll           |
| Zoom at cursor               | Ctrl + Scroll            |
| Pan (free)                   | Middle-click + drag      |

---

## Output Format

### YOLO `.txt` — separate directories for training compatibility

Annotations are saved to two subdirectories so each is directly compatible with
Ultralytics `yolo detect train` and `yolo segment train`:

```
images/
├── img001.jpg
├── img002.jpg
└── labels/
    ├── detect/
    │   ├── img001.txt      # bounding boxes only
    │   └── img002.txt
    └── segment/
        ├── img001.txt      # polygon masks only
        └── img002.txt
```

**Detection** (`labels/detect/`):

```
<class_id> <x_center> <y_center> <width> <height>
```

**Segmentation** (`labels/segment/`):

```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

All values are **normalized to 0–1** relative to image dimensions.

### CSV (consolidated `annotations.csv` in image folder)

```
image, type, class_id, class_name, x1, y1, x2, y2, polygon_points
```

Absolute pixel coordinates for convenience. Includes both box and polygon entries.
Polygon bounding boxes are auto-computed from vertices. Rebuilt from all label
files on exit.

### `classes.txt`

When you add classes via the toolbar, a `classes.txt` file is saved in the image
folder. This file is automatically loaded on next launch so class names persist.

---

## Features

- **Dark theme** — modern CustomTkinter UI
- **Box + Polygon modes** — toggle with `m` key or toolbar button
- **Vertex streaming** — continuous vertex placement while moving the mouse (`v` to toggle)
- **Edge snapping** — snap to nearby polygon edges while streaming (`s` to toggle)
- **Polygon selection** — click a polygon to select it for editing; Escape to deselect
- **Full vertex editing** — drag, insert on edge, right-click delete (on selected polygon)
- **Snapshot undo / redo** — `Ctrl+Z` / `Ctrl+Y` undoes/redoes any mutation including deletes, vertex moves, and edge inserts
- **Multi-class support** — dropdown selector + inline "Add" for new classes, per-class colors
- **Completion tracking** — mark images as complete with a checkbox; filter by status (All / Complete / Partial / Unannotated)
- **Annotation stats** — per-image and per-session timing, annotation counts, saved to `annotation_stats.json`
- **Separate label dirs** — `labels/detect/` and `labels/segment/` for clean Ultralytics training
- **Dynamic symbology** — line widths, vertex sizes, and labels scale with zoom level
- **Fit-to-view** — auto-fits image on open and window resize
- **Wrap-around navigation** — next/prev image wraps to start/end instead of closing
- **EXIF orientation** — auto-corrects rotated phone photos
- **Viewport cropping** — only renders the visible region, safe at any zoom level
- **Save on navigate** — annotations are saved when you change images, quit, or close the window
- **Open Folder button** — switch image folders mid-session
- **Consolidated CSV** — pixel-coordinate CSV export rebuilt from all label files on exit

---

## Authors

Nathan Miller · Zack Loken

---

## License

[MIT](LICENSE)
