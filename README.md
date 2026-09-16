# YoloLabeler

> Published as **YOLO Annotator**.

Desktop tool for drawing and reviewing YOLO bounding-box and instance-segmentation
annotations, and for visualizing model predictions against ground truth.
Built with Python + CustomTkinter.

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

To run the test suite:

```bash
pip install -e .[test]
pytest -q
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

A folder dialog will open; select a folder of images.

**Or pass a folder directly:**

```bash
yololabeler /path/to/images
```

**Or run as a module:**

```bash
python -m yololabeler /path/to/images
```

### Running the packaged build

A Windows build needs no Python or conda install. `packaging/yololabeler.spec`
builds a PyInstaller windowed, one-directory bundle; the result is
`packaging/dist/YoloLabeler/YoloLabeler.exe`, alongside its supporting files.
Build instructions are in `packaging/README.md`.

The executable is unsigned, so the first run shows a Windows SmartScreen prompt
("Windows protected your PC"). Choose "More info", then "Run anyway" to continue.

---

## Controls

One workspace: annotate and review on the same canvas. Press `h` in the app
for the same list, filtered to the current mode.

<!-- controls:start -->
| Action | Key |
|---|---|
| Previous image | `Left` |
| Next image | `Right` |
| Previous queue item | `Up` |
| Next queue item | `Down` |
| Accept focused item | `a` |
| Reject focused item | `r` |
| Select the focused item's annotation for editing | `e` |
| Fit image to window | `f` |
| Zoom to focused item | `z` |
| Toggle box / polygon mode | `m` |
| Toggle vertex snapping | `s` |
| Toggle vertex streaming | `v` |
| Select class by id | `0`-`9` |
| Rename the active class | `Ctrl+R` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Save now | `Ctrl+S` |
| Left click at the cursor | `Space` |
| Cancel polygon / deselect | `Escape` |
| Toggle this help | `h` |

| Action | Input | Mode |
|---|---|---|
| Zoom at cursor | Ctrl+Scroll | always |
| Pan up / down | Scroll | always |
| Pan left / right | Shift+Scroll | always |
| Pan | Middle-click drag | always |
| Draw a box | Left-click drag | box |
| Delete box | Right-click | box |
| Place vertex / select polygon | Left-click | polygon |
| Close polygon | Double-click | polygon |
| Move vertex (selected polygon) | Drag vertex | polygon |
| Insert vertex (selected polygon) | Click edge | polygon |
| Delete vertex / polygon | Right-click | polygon |
<!-- controls:end -->

Vertex streaming (`v`) places vertices continuously as the mouse moves instead of
one per click: click to start streaming, move the mouse to trace the outline, click
again to pause, and double-click to finish the polygon; `Escape` cancels it instead.
Toggle edge snapping (`s`) at the same time to pull streamed vertices onto nearby
existing edges.

---

## Folder Structure

```
images/
├── img001.jpg
├── img002.jpg
├── state/
│   ├── annotation_stats.json
│   ├── review_stats.json
│   ├── classes.json
│   └── annotations/
│       ├── img001.json         # per-annotation id, author, source, provenance
│       └── img002.json
├── labels/
│   ├── detect/
│   │   ├── img001.txt          # bounding boxes (YOLO format)
│   │   ├── img002.txt
│   │   └── .original/          # backup made before the first destructive edit
│   └── segment/
│       ├── img001.txt          # polygon masks (YOLO format)
│       ├── img002.txt
│       └── .original/
└── predictions/
    ├── manifest.json           # model name and import provenance
    ├── detect/
    │   ├── img001.txt          # model-predicted boxes
    │   └── img002.txt
    └── segment/
        ├── img001.txt          # model-predicted polygons
        └── img002.txt
```

`state/annotations/<stem>.json` is a sidecar next to the label files: the label
lines stay the geometry of record, and the sidecar carries id, author, creation
time and provenance (`drawn`, `accepted`, or `unknown` for a hand-edited line with
no matching record) per annotation, joined to its label line by the line's exact
formatted text and its position among identical lines.
`predictions/manifest.json` and `labels/*/.original/` are written the first time
Import predictions runs and the first time a label file is saved, respectively;
neither exists until then.

---

## Output Formats

### YOLO `.txt`: separate directories for training compatibility

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

### Predictions

`predictions/detect/` and `predictions/segment/` hold model output in the same
YOLO format as labels, plus a confidence column. This canonical layout is the
only format the application reads directly:

Detection: `<class_id> <confidence> <x_center> <y_center> <width> <height>`

Segmentation: `<class_id> <confidence> <x1> <y1> ... <xN> <yN>`

All values are normalized 0-1, same as labels, with confidence in column two.
`predictions/manifest.json`, written by Import predictions, records where a
dataset's predictions came from:

```json
{"model": "nathan_v15", "source_format": "bur_detect_json", "source_dir": "...",
 "imported_at": "...", "imported_by": "zack", "class_id_default": 0, "files": 340}
```

`source_dir` is the source folder that import read, so a dataset's predictions can
be traced back to the folder they were converted from.

When no manifest is present and prediction text files already exist (for example,
a dataset from before this format existed), they are read as-is; the manifest is
written by Import predictions, not required for the files to be read.

Predictions are matched against ground truth using IoU (default 0.60, a fixed
constant) to classify each as a false positive, a false negative (a ground-truth
annotation with no matching prediction, called a model miss), or a true positive;
see [Reviewing predictions](#reviewing-predictions).

### Import predictions

The Import predictions toolbar button, next to Open Folder, opens a form: source
folder, source format, a class id, and a model name. It converts the chosen
format into the canonical layout above and writes the manifest.

| Format | Source | Class id | Notes |
|---|---|---|---|
| `yololabeler` | This tool's own canonical prediction layout | read from the source file | revalidated and rewritten in the same layout |
| `ultralytics_txt` | Ultralytics `save_txt(save_conf=True)` output, confidence last: detect `class cx cy w h conf`, segment `class x1 y1 ... xn yn conf` | read from the source file | confidence is moved to column two; a line with no confidence value is rejected and counted rather than kept |
| `bur_detect_json` | Nathan's `bur_detect.py` output, `{"boxes": [[x0,y0,x1,y1]], "scores": [...]}` in full-image pixels, one file per image stem | required on the form, since the JSON carries none | normalized by each image's oriented size |

The class id field only applies to `bur_detect_json`; the other two formats carry
their own class ids and ignore it. Import parses every source file before writing
anything, then reports one line on the canvas: files written, files skipped (no
matching image in the folder), and lines rejected. If any image in the folder
carries an EXIF rotation, the summary adds a count of affected images, since
Nathan's script does not transpose and the mismatch can otherwise land boxes
rotated; coordinates are never transformed to correct for this automatically.

### Batch import: `yololabeler-import`

The toolbar form imports one source folder into the image folder that is open.
When a collaborator hands over a whole tree of prediction folders mirroring a
whole tree of image folders, `yololabeler-import` converts all of them in one
pass. It is a separate console script with no Tk or CustomTkinter import, so it
runs on a headless machine.

```bash
yololabeler-import /path/to/predictions /path/to/images \
  --format bur_detect_json --model nathan_v15 --class-id 0 --user zack
```

That is a dry run: it writes nothing. Add `--write` to actually import. A real
run is never the default, since a batch job touches every image folder at once.

Folders pair strictly by their path relative to their own root:
`predictions/Farm/Field/Mission3` imports into `images/Farm/Field/Mission3` and
nowhere else. A leaf folder name that matches under a different parent is not a
match, and source files are never indexed by stem across folders, so two missions
holding a `DJI_0001` cannot cross over into each other.

The dry run prints, before any write could happen:

- the matched pairs, with each pair's source file count and image count
- source folders with no image folder at the same relative path
- image folders with no source folder at the same relative path
- source stems found under more than one source folder

Unpaired folders are reported, never guessed at or fuzzy-matched; resolving them
is a decision for whoever knows the dataset. The stem list is the warning that
matters most: when the same stem appears in several source folders, a mispairing
writes plausible-looking boxes from the wrong flight and nothing errors.

A `--write` run prints the same per-folder summary the toolbar form shows on the
canvas, one line per pair, then a rollup of total images written, files skipped
and lines rejected. `--class-id` is required for `bur_detect_json` and ignored by
the other two formats, same as the form. The source root and the images root may
not be the same folder or nested inside one another. Nothing is ever deleted
from the source tree; removing it afterwards is a manual step.

Reach for the toolbar form for a single folder you are about to annotate, and for
`yololabeler-import` for a handover of many folders at once.

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

One workspace: annotate and review on the same canvas, no tab switch to lose a
viewport or leave unsaved work behind.

- Box + Polygon modes: toggle with `m` or the toolbar button
- Vertex streaming and edge snapping: continuous vertex placement while moving the
  mouse (`v`), snapped to nearby edges (`s`)
- Full vertex editing: drag, insert on an edge, right-click delete, on the
  selected polygon; snapshot undo / redo (`Ctrl+Z` / `Ctrl+Y`) covers drawing,
  accept, reject and edit alike
- Multi-class support: dropdown selector, inline "Add" for new classes, per-class
  colors
- Prediction layer: import model output, matched against ground truth by IoU, and
  drawn as a dashed overlay in a lightened tint of the class colour, with the
  focused item highlighted
- Accept in place: promote an unmatched prediction straight into an annotation
  with its geometry, class and provenance recorded, no separate review pass
- Queue over every image: unmatched predictions, model misses and matches,
  stepped through with `Up` / `Down`, filterable by class, type and status
- Blind images: a per-image flag that hides predictions until the image is marked
  Complete, for measuring assisted versus unassisted annotation
- Completion tracking: mark an image Complete to record who, when, how many
  annotations, and which model; filter the image list by status
- Import predictions: one form converts three source formats, this tool's own
  layout, Ultralytics `save_txt(save_conf=True)` output, and Nathan's
  `bur_detect.py` JSON, into the canonical prediction layout and a manifest
- Canvas banner: import results, migrated data counts, rejected label or
  prediction lines, and save failures are reported in one message on the canvas;
  no pop-ups for routine feedback
- Annotation stats: per-image and per-session timing, annotation counts
  (`annotation_stats.json`)
- Separate label dirs: `labels/detect/` and `labels/segment/` for clean
  Ultralytics training
- Dynamic symbology: line widths, vertex sizes, and labels scale with zoom level
- Text halo: labels use dark outlines for readability on any background
- Fit-to-view: auto-fits image on open and window resize
- EXIF orientation: auto-corrects rotated phone or drone photos
- Viewport cropping: only renders the visible region, safe at any zoom level
- Save on navigate: `go_to_image` saves the current image before loading the
  next; a failed save on quit prompts once, with a count of what would be lost
- Original label backup: `.original/` copies made the first time a label file
  is saved
- Packaged build: a PyInstaller windowed build for Windows needs no Python
  install, see [Running the packaged build](#running-the-packaged-build)

---

## Reviewing predictions

There is one canvas: predictions load for every image that is not blind, get
matched against the ground-truth annotations, and show up as a queue of items
alongside the annotation tools, no separate mode to enter.

### The queue

Every prediction above the confidence threshold is matched against ground-truth
annotations of the same class, all candidate pairs scored by IoU and assigned
greedily, highest-IoU first, so each annotation and each prediction participates
in at most one match. The queue is the flattened result, in this fixed order:
unmatched predictions, then model misses (a ground-truth annotation no
prediction matched), then matches. The Class, Type and Status dropdowns in the
status bar filter it; Status filters on verdict presence, so "Not reviewed" means
no verdict yet, whatever the type. True-positive, false-positive and
false-negative counts for the current image are shown in the status bar.

### Stepping

`Up` / `Down` move the focus one queue item at a time, wrapping at both ends.
Stepping sets the active class and mode (box or polygon) to match the focused
item, so it is always drawn under the same visibility rule the canvas already
uses, then zooms so the item fills roughly one third of the canvas. `z` re-zooms
to the current item without moving the focus. The GT and Pred checkboxes toggle
the annotation and prediction overlays independently; only the focused item
carries a label, everything else is an unlabelled outline.

### Accept and reject

| Queue item | Accept (`a`) | Reject (`r`) |
|---|---|---|
| Unmatched prediction | Insert an annotation with the prediction's geometry and class (`source: accepted`); verdict `accepted` | Verdict `rejected`; nothing else changes |
| Match (prediction paired with an annotation) | Verdict `accepted`; annotation unchanged | Delete the annotation; verdict `rejected` |
| Model miss (annotation with no prediction) | Verdict `accepted` | Delete the annotation; verdict `rejected` |

Press `e` to select the focused item's paired annotation for editing, moving or
deleting its vertices without leaving the queue. Every accept, reject and edit is
one undo step, covered the same way as drawing (`Ctrl+Z` / `Ctrl+Y`).

### Threshold

Predictions below the confidence threshold are neither drawn nor matched. It
defaults to 0.50, is stored per dataset in `state/review_stats.json`, and is
shown and edited in the status bar's Conf entry; press Enter to apply, an
out-of-range or non-numeric value reverts to the stored one. The IoU threshold is
a fixed 0.60, shown next to it.

### Blind images

Ticking Blind on an image stops its prediction files from being read: the strip
shows "Blind" in place of the queue and counts, and Accept / Reject are disabled.
Ticking Complete on a blind image records the completion with `blind: true`;
predictions load normally afterward, and any later accept still carries
`source: accepted`, so the blind pass and the assisted pass stay distinguishable
in the sidecar.

### Completion

Complete is the one dataset gate: ticking it writes a completion record and
saves; unticking it removes the record. If queue items are still missing a
verdict the checkbox label reads "Complete (n not reviewed)", but the tick is
never blocked on it.

### What is written where

`state/review_stats.json`: `settings.conf_threshold`; per image, a `verdicts`
dict keyed by prediction id (a model miss is keyed by its annotation's id
instead), each verdict recording `action` (`accepted` / `rejected`),
`kind` (`fp` / `fn` / `tp`), `class_id`, `conf`, `iou`, `by` and `at`;
a `labels_backed_up` flag set once the first `.original/` backup is made.

`state/annotation_stats.json`: a `completion` entry per completed image,
`{"by", "at", "blind", "annotation_count", "model"}`, where `model` is the name
recorded in the predictions manifest when predictions were visible, or `null`
for a blind pass.

---

## Authors

Zack Loken

---

## License

[MIT](LICENSE)
