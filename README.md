# YoloLabeler

### Annotate Tab
![Annotate Tab](GUI_annotate.png)

### Review Tab
![Review Tab](GUI_review.png)

Desktop tool for drawing and reviewing YOLO bounding-box and instance-segmentation
annotations, and for visualizing model predictions against ground truth.
Built with Python + CustomTkinter; no GPU, no server, no browser required.

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
- Shapely ≥ 2.0

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


---

## Controls

### Annotate tab

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

### Review tab

| Action                        | Input  |
|-------------------------------|--------|
| Accept detection              | `a`    |
| Reject detection              | `r`    |
| Edit detection (→ Annotate)   | `e`    |
| Previous / Next detection     | `←` / `→` |
| Next / Previous image         | `↑` / `↓` |
| Toggle Box / Polygon mode     | `m`    |
| Toggle help overlay           | `h`    |

Press `h` in either tab for a full keybinding reference including per-action
behavior for FP / FN / TP detections.

---

## Folder Structure

```
images/
├── img001.jpg
├── img002.jpg
├── state/
│   ├── annotation_stats.json
│   ├── review_stats.json
│   └── classes.json
├── labels/
│   ├── detect/
│   │   ├── img001.txt          # bounding boxes (YOLO format)
│   │   └── img002.txt
│   └── segment/
│       ├── img001.txt          # polygon masks (YOLO format)
│       └── img002.txt
└── predictions/
    ├── detect/
    │   ├── img001.txt          # model-predicted boxes
    │   └── img002.txt
    └── segment/
        ├── img001.txt          # model-predicted polygons
        └── img002.txt
```

---

## Output Formats

### YOLO `.txt` — separate directories for training compatibility

Annotations are saved to two subdirectories so each is directly compatible with
Ultralytics `yolo detect train` and `yolo segment train`.

**Detection** (`labels/detect/`):

```
<class_id> <x_center> <y_center> <width> <height>
```

**Segmentation** (`labels/segment/`):

```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

All values are **normalized to 0–1** relative to image dimensions.

### Predictions (for review)

Place model prediction files in `predictions/detect/` and `predictions/segment/`
using the same YOLO format with an added confidence score:

**Detection**: `<class_id> <confidence> <x_center> <y_center> <width> <height>`

**Segmentation**: `<class_id> <confidence> <x1> <y1> ... <xN> <yN>`

The Review tab matches predictions against ground truth using IoU (default 0.60)
to classify each as TP, FP, or FN.

### `classes.json`

When you add classes via the toolbar, a `classes.json` file is saved in the image
folder. This file stores class names and colors and is automatically loaded on
next launch.

```json
{
  "0": {"name": "catkin", "color": "#e6194b"},
  "1": {"name": "bud", "color": "#3cb44b"}
}
```

---

## Features

### Annotate tab
- **Box + Polygon modes:** toggle with `m` key or toolbar button
- **Vertex streaming:** continuous vertex placement while moving the mouse (`v` to toggle)
- **Edge snapping:** snap to nearby polygon edges while streaming (`s` to toggle)
- **Polygon selection:** click a polygon to select it for editing; Escape to deselect
- **Full vertex editing:** drag, insert on edge, right-click delete (on selected polygon)
- **Snapshot undo / redo:** `Ctrl+Z` / `Ctrl+Y` for any mutation
- **Multi-class support:** dropdown selector + inline "Add" for new classes, per-class colors
- **Completion tracking:** mark images as complete; filter by status
- **Annotation stats:** per-image and per-session timing, annotation counts (`annotation_stats.json`)
- **Separate label dirs:** `labels/detect/` and `labels/segment/` for clean Ultralytics training
- **Dynamic symbology:** line widths, vertex sizes, and labels scale with zoom level
- **Text halo:** annotation labels use dark outlines for readability on any background
- **Fit-to-view:** auto-fits image on open and window resize
- **EXIF orientation:** auto-corrects rotated phone photos
- **Viewport cropping:** only renders the visible region, safe at any zoom level
- **Save on navigate:** annotations are saved when you change images, quit, or close

### Review tab
- **IoU-based matching:** automatically matches predictions to ground truth (IoU ≥ 0.60)
- **Detection cycling:** step through FP / FN / TP detections with auto-zoom
- **Accept / Reject / Edit:** per-detection actions with type-specific behavior
- **Prediction reference overlay:** dashed blue overlay shows prediction geometry while editing
- **Viewport sync:** zoom and position carry over between Annotate and Review tabs
- **Review state persistence:** progress saved to `review_stats.json`, survives across sessions
- **Review timer:** tracks time spent reviewing per image
- **Original label backup:** `.original/` copies made before first destructive edit

---

## Authors

Nathan Miller · Zack Loken

---

## License

[MIT](LICENSE)
