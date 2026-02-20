# BoxLabeler

Lightweight desktop tool for drawing YOLO bounding-box annotations on images.
Built with Python + Tkinter — no GPU, no server, no browser required.

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Install

```bash
pip install git+https://github.com/loken-usda/yolo-annotator.git
```

Or for development:

```bash
git clone https://github.com/loken-usda/yolo-annotator.git
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
boxlabeler
```

A folder dialog will open — select a folder of images.

**Or pass a folder directly:**

```bash
boxlabeler /path/to/images
```

**Or run as a module:**

```bash
python -m boxlabeler /path/to/images
```

---

## Controls

| Action                  | Input                    |
|-------------------------|--------------------------|
| Draw a box              | Left-click + drag        |
| Delete a box            | Right-click on box       |
| Next / Previous image   | Right / Left arrow       |
| Pan up / down           | Scroll                   |
| Pan left / right        | Shift + Scroll           |
| Zoom at cursor          | Ctrl + Scroll            |
| Pan (free)              | Middle-click + drag      |
| Undo last box           | `z`                      |
| Toggle help overlay     | `h`                      |
| Quit (saves first)      | `Escape` or close window |

---

## Output Format

### YOLO `.txt` (one per image, in `labels/` subdirectory)

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalized to 0–1** relative to image dimensions.  
Compatible with Ultralytics YOLOv5/v8/v11, Darknet, and most YOLO training pipelines.

### CSV (one per image, in `labels/` subdirectory)

```
class_id, class_name, x1, y1, x2, y2
```

Absolute pixel coordinates for convenience.

### `classes.txt`

When you add classes via the toolbar, a `classes.txt` file is saved in the image
folder. This file is automatically loaded on next launch so class names persist.

---

## Features

- **Multi-class support** — dropdown selector + inline "Add" for new classes
- **Auto-resume** — opens at the first unlabeled image
- **EXIF orientation** — auto-corrects rotated phone photos
- **Viewport cropping** — only renders the visible region, safe at any zoom level
- **Save on exit** — annotations are saved when you quit or close the window
- **Open Folder button** — switch image folders mid-session
- **Progress counter** — toolbar shows labeled count vs total

---

## Authors

Nathan Miller · Zack Loken

---

## License

[MIT](LICENSE)
