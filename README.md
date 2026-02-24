# YoloLabeler

![YoloLabeler GUI](GUI.png)

Lightweight desktop tool for drawing YOLO bounding-box and instance segmentation
annotations on images.
Built with Python + Tkinter — no GPU, no server, no browser required.

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
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

- Python 3.8+
- Pillow ≥ 9.0
- Tkinter (included with most Python installations)

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
|                      **Box mode**                       |
| Draw a box                   | Left-click + drag        |
| Delete a box                 | Right-click on box       |
|                     **Polygon mode**                    |
| Place vertex                 | Left-click               |
| Close polygon                | Double-click             |
| Move vertex                  | Drag vertex              |
| Insert vertex on edge        | Click on edge            |
| Delete vertex                | Right-click vertex       |
| Delete polygon               | Right-click in polygon   |
| Cancel polygon in progress   | `Escape`                 |
|                  **Navigation & view**                  |
| Next / Previous image        | Right / Left arrow       |
| Pan up / down                | Scroll                   |
| Pan left / right             | Shift + Scroll           |
| Zoom at cursor               | Ctrl + Scroll            |
| Pan (free)                   | Middle-click + drag      |
| Select class by id           | `0`–`9`                  |
| Undo                         | `Ctrl+Z`                 |
| Redo                         | `Ctrl+Y`                 |
| Toggle help overlay          | `h`                      |

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

- **Box + Polygon modes** — toggle with `m` key or toolbar button
- **Full vertex editing** — drag, insert on edge, right-click delete
- **Multi-class support** — dropdown selector + inline "Add" for new classes
- **Undo / Redo** — `Ctrl+Z` / `Ctrl+Y` for boxes, polygons, and vertices
- **Separate label dirs** — `labels/detect/` and `labels/segment/` for clean Ultralytics training
- **Auto-resume** — reopens at the last annotated image
- **EXIF orientation** — auto-corrects rotated phone photos
- **Viewport cropping** — only renders the visible region, safe at any zoom level
- **Save on navigate** — annotations are saved when you change images, quit, or close the window
- **Open Folder button** — switch image folders mid-session
- **Progress counter** — toolbar shows number of labeled images and annotation class counts

---

## Authors

Nathan Miller · Zack Loken

---

## License

[MIT](LICENSE)
