# YoloLabeler

> Published as **YOLO Annotator**.

Desktop tool for drawing and reviewing YOLO bounding-box and instance-segmentation
annotations, and for visualizing model predictions against ground truth.
Built with Python + CustomTkinter.

![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)
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

- Python 3.14+
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
| Previous item | `Down` |
| Next item | `Up` |
| Accept the focused item; an FP becomes an annotation | `a` |
| Reject the focused item and delete its annotation | `r` |
| Edit the focused item; an FP is accepted first | `e` |
| Comment on the focused item and flag it | `c` |
| Fit the image to the window | `f` |
| Zoom to the focused item | `z` |
| Switch between box and polygon mode | `m` |
| Vertex snapping on or off | `s` |
| Vertex tracing on or off | `t` |
| Pick a class by id | `0`-`9` |
| Rename the active class | `Ctrl+R` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Save now | `Ctrl+S` |
| Discard the polygon in progress | `Escape` |
| Show or hide this help | `h` |

| Action | Input | Mode |
|---|---|---|
| Zoom at the cursor | Ctrl+Scroll | always |
| Pan up or down | Scroll | always |
| Pan left or right | Shift+Scroll | always |
| Pan | Middle-click drag | always |
| Select the shape | Click an outline | always |
| Move the shape | Drag an outline | always |
| Delete the shape | Right-click an outline | always |
| Draw a box, even inside another | Left-click drag | box |
| Resize the selected box from the opposite corner | Drag a corner | box |
| Place a vertex, even inside another polygon | Left-click | polygon |
| Start or pause tracing vertices | Left-click (Trace on) | polygon |
| Close the polygon | Double-click | polygon |
| Discard the polygon in progress | Right-click | polygon |
| Start a new polygon on that vertex | Click a vertex | polygon |
| Select its polygon instead | Alt+click a vertex | polygon |
| Move a vertex of the selected polygon | Drag a vertex | polygon |
| Insert a vertex in the selected polygon | Shift+click an edge | polygon |
| Delete a vertex | Right-click a vertex | polygon |
| Open or close the legend | Click Legend | always |
<!-- controls:end -->

The mouse follows one rule for both shapes: what is under the pointer decides
what a press does. On an outline, a click selects the shape, a drag moves it
whole, and a right-click deletes it. On a handle of the selected shape, a box
corner or a polygon vertex, a drag reshapes it. On empty canvas, box mode drags
out a box and polygon mode places a vertex, even inside another shape, e.g. for a
bur that sits inside its neighbour's outline. A click on any polygon vertex starts
a new polygon on that vertex, selected or not, so a shared boundary begins on the
neighbour's own point; `Alt`+click selects that polygon instead. `Shift`+click on
the selected polygon's edge inserts a vertex there and drags it. `Escape` has one
job, discarding the polygon in progress; deselecting is a click on empty canvas.

Vertex tracing (`t`) places vertices continuously as the mouse moves instead of
one per click: click to start tracing, move the mouse along the outline, click
again to pause, and double-click to finish the polygon; `Escape` or a right-click
discards it instead. A new vertex is laid each time the pointer moves 15 screen
pixels, so the spacing looks the same at every zoom, and the polygon you get is
exactly the trace you watched form; the start, pause and resume clicks place
exact vertices, so a corner is a click.
Snapping (`s`) pulls a placed or traced vertex onto an existing vertex within 15
screen pixels, never onto a point along an edge, so a shared boundary reuses the
neighbour's own vertices. A dashed ring around the pointer, with a dot on the
vertex itself, marks the vertex a click would snap to. A polygon closed with no
area, or a vertex drag or delete that would flatten one, is refused; the shape
keeps its previous geometry.

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
`yololabeler-import` runs and the first time a label file is saved, respectively;
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
`predictions/manifest.json`, written by `yololabeler-import`, records where a
dataset's predictions came from:

```json
{"model": "nathan_v15", "source_format": "bur_detect_json", "source_dir": "...",
 "imported_at": "...", "imported_by": "zack", "class_id_default": 0, "files": 340}
```

`source_dir` is the source folder that import read, so a dataset's predictions can
be traced back to the folder they were converted from.

When no manifest is present and prediction text files already exist (for example,
a dataset from before this format existed), they are read as-is; the manifest is
written by `yololabeler-import`, not required for the files to be read.

Predictions are matched against ground truth using IoU (default 0.50, a fixed
constant) to classify each as a false positive, a false negative (a ground-truth
annotation with no matching prediction, called a model miss), or a true positive;
see [Reviewing predictions](#reviewing-predictions).

### Batch import: `yololabeler-import`

`yololabeler-import` converts predictions before you open a folder. It takes a
source format, a source folder, and a model name, and converts the chosen
format into the canonical layout above, writing the manifest.

| Format | Source | Class id | Notes |
|---|---|---|---|
| `yololabeler` | This tool's own canonical prediction layout | read from the source file | revalidated and rewritten in the same layout |
| `ultralytics_txt` | Ultralytics `save_txt(save_conf=True)` output, confidence last: detect `class cx cy w h conf`, segment `class x1 y1 ... xn yn conf` | read from the source file | confidence is moved to column two; a line with no confidence value is rejected and counted rather than kept |
| `bur_detect_json` | Nathan's `bur_detect.py` output, `{"boxes": [[x0,y0,x1,y1]], "scores": [...]}` in full-image pixels, one file per image stem | required via `--class-id`, since the JSON carries none | normalized by each image's oriented size |

The class id field only applies to `bur_detect_json`; the other two formats carry
their own class ids and ignore it. Import parses every source file before writing
anything, then reports files written, files skipped (no matching image in the
folder), and lines rejected. If any image in the folder carries an EXIF rotation,
the summary adds a count of affected images, since Nathan's script does not
transpose and the mismatch can otherwise land boxes rotated; coordinates are
never transformed to correct for this automatically.

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

A `--write` run prints one line per pair, then a rollup of total images written,
files skipped and lines rejected. `--class-id` is required for `bur_detect_json`
and ignored by the other two formats. The source root and the images root may
not be the same folder or nested inside one another. Nothing is ever deleted
from the source tree; removing it afterwards is a manual step.

Use `yololabeler-import` to convert predictions before opening a folder; there
is no in-app import form.

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
- Vertex tracing and vertex snapping: continuous vertex placement while moving the
  mouse (`t`), snapped to nearby existing vertices (`s`)
- Full vertex editing: drag a vertex, Shift+click an edge to insert one, and
  right-click delete on the selected polygon; both kinds are selected, moved and
  right-click deleted by their outline, never their interior, so a press inside
  a shape draws a new one, e.g. for a bur that sits inside its neighbour's; a box
  also resizes from a corner; hovering an outline shows its handles; snapshot
  undo / redo (`Ctrl+Z` / `Ctrl+Y`) covers drawing, accept, reject and edit alike
- Multi-class support: dropdown selector, inline "Add" for new classes, per-class
  colors
- Prediction layer: import model output, matched against ground truth by IoU, and
  drawn as a dashed overlay outlined by review status, labelled in class colour,
  with the focused item highlighted
- Accept in place: promote an unmatched prediction straight into an annotation
  with its geometry, class and provenance recorded, no separate review pass
- Queue over every image: unmatched predictions, model misses and matches,
  stepped through with `Up` / `Down`, filterable by class, type and status
- Blind images: a per-image flag that hides predictions until the image is marked
  Completed, for measuring assisted versus unassisted annotation
- Completion tracking: mark an image Completed to record who, when, how many
  annotations, and which model; filter the image list by status
- Canvas banner: migrated data counts, rejected label or prediction lines, and
  save failures are reported in one message on the canvas; no pop-ups for
  routine feedback
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
annotations of the same class and kind (a box prediction never matches a
polygon annotation, or the reverse), all candidate pairs scored by IoU and assigned
greedily, highest-IoU first, so each annotation and each prediction participates
in at most one match. The queue is the flattened result: unmatched predictions,
model misses (a ground-truth annotation no prediction matched) and matches,
together in one spatial path. The path starts at the item nearest the top-left
corner and always steps to the nearest item not yet visited, so a cluster is
reviewed in a row whatever the types in it. The Type and Class filters lay the
path over the items they keep; the Review status filter only hides items, so
judging one never reorders the rest. After an accept or reject the focus moves
to the next unreviewed item along the path. Class selection lives in the top toolbar
alongside the drawing controls, with an "All" option; Type sits in the status
bar's left group and Review status in its right group, by the item stepper. Choosing a class filters the queue and limits the
canvas to that class; "All" lifts the class limit so every class is drawn, but
not the mode limit, so Box mode still draws only boxes. Drawing needs a class,
so it is blocked while "All" is selected. Review status filters on verdict presence, so "Not
reviewed" means no verdict yet, whatever the type. (Image status, in the top
toolbar, is the separate per-image Completed / Partial / Unannotated filter.) An
annotation you draw yourself is recorded as `accepted` the moment it is added,
so it never waits in the queue for a review action. True-positive,
false-positive and false-negative counts for the current image sit in the
status bar's left group, after the filters they describe. The window title
carries the rest: the image name, the zoom, the time spent on this image and
the current user, e.g. "YoloLabeler - img001.jpg - 400% - 0:16 - zack".

### Stepping

`Up` moves the focus to the next queue item and `Down` to the previous one,
wrapping at both ends.
Stepping sets the active class and mode (box or polygon) to match the focused
item, so it is always drawn under the same visibility rule the canvas already
uses, then zooms so the item fills roughly one third of the canvas. `z` re-zooms
to the current item without moving the focus. The Labels checkbox on the top
toolbar and the Predictions checkbox in the status bar toggle the annotation and
prediction overlays independently; turning Labels off hides every annotation
shape, focused or not. The item stepper sits in the status bar's right end, built
to the same spec as the image stepper at the right end of the toolbar: Prev, a
typable position entry, the queue total, Next. Typing a position and pressing
`Enter` focuses that item; anything out of range reverts. The item's type and
verdict are not repeated here, since the placard at the canvas's top right
already carries both.
Left of the stepper sit the Review status filter and the Reviewed box, which
lights when the focused item has a verdict; unticking it clears that verdict
back to not reviewed. Unticking does not undo what accept or reject did to the
annotation, only the judgement, so use `Ctrl+Z` to take back a deletion.

Annotations are solid and predictions dashed at the same zoom-scaled width, so
the line style says whose shape it is. On an image that has predictions, both
are outlined by review status: green accepted, orange not reviewed, red
rejected. A match's prediction and annotation share one status. Unticking
Predictions hides the model's boxes but leaves the annotations in their status
colours; only a blind image or an image with no predictions puts annotations
back in their class colour. The class dropdown narrows predictions to the
chosen class the same way it narrows annotations, with the focused item drawn
whatever its class. A shape's label text is
always drawn in its class colour, whichever colour the outline carries, so class
identity stays readable during review. The focused item is
marked by a highlighter-blue glow under every shape it has, its prediction and
its annotation alike, and the one thing that is selected
for editing is drawn in that blue with vertex or corner handles. Every annotation
carries a "class: name" label; a prediction is labelled, with its confidence,
only when it is the focused item or an FP, since a TP's label already sits on
its annotation. The Legend chip in the lower left of
the canvas opens a key to all of this; click it again to close it.

### Accept and reject

| Queue item | Accept (`a`) | Reject (`r`) |
|---|---|---|
| Unmatched prediction | Insert an annotation with the prediction's geometry and class (`source: accepted`); verdict `accepted` | Verdict `rejected`; nothing else changes |
| Match (prediction paired with an annotation) | Verdict `accepted`; annotation unchanged | Delete the annotation; verdict `rejected` |
| Model miss (annotation with no prediction) | Verdict `accepted` | Delete the annotation; verdict `rejected` |

Accept and reject both move straight on to the next item without a review
action; the accepted prediction stays drawn, dashed, under its new annotation.
Once every item of the active Type and Class filters has a verdict, the image
stays loaded and zooms out to 33%, centred, for a pass over the whole canopy for
objects the model missed.

Edit (`e`, or the Edit button between Accept and Reject) selects the focused
item's annotation for editing, moving or deleting its vertices without leaving
the queue. An unmatched prediction has no annotation yet, so Edit accepts it
first and selects the annotation that creates, keeping the focus on it. Every accept, reject and edit is
one undo step, covered the same way as drawing (`Ctrl+Z` / `Ctrl+Y`).
Dragging the focused item's GT box updates its displayed classification, IoU and
reviewed status immediately, with no separate action and no new verdict; how far
an edited box has moved from the prediction it came from is read by comparing its
geometry to that prediction (`prediction_id` in the sidecar), not from a verdict.

### Threshold

Predictions below the confidence threshold are neither drawn nor matched. It
defaults to the lowest confidence in the imported set (`min_conf` in
`predictions/manifest.json`, which `yololabeler-import` writes), and to 0.25 only
when there is no manifest. It is shown and edited in the status bar's Conf entry;
press Enter to apply, an out-of-range or non-numeric value reverts to the stored
one. A typed value is stored in `state/review_stats.json` with the manifest's
`imported_at`, so re-running the import resets it to the new set's minimum. The IoU threshold is
a fixed 0.50, neither shown nor editable in the UI.

### Blind images

Blind pass sits in the status bar next to the Predictions checkbox, since both
govern whether the model's boxes are in play. Ticking it on an image stops its
prediction files from being read: the
strip empties the item stepper to `/ 0` and clears the counts, and Accept / Edit
/ Reject are disabled. The Type/Review status filters, Conf entry, and the item
stepper are greyed out for the same reason: none of them do anything until
predictions are back.
Ticking Completed on a blind image records the completion with `blind: true`;
predictions load normally afterward, and any later accept still carries
`source: accepted`, so the blind pass and the assisted pass stay distinguishable
in the sidecar.

### Flagging for a second look

`c` opens a comment box for the focused item, prediction or annotation alike.
Enter saves it and flags the item, with or without a comment; a flag is
independent of the verdict, so an item can be flagged before it is judged or
after. A flagged item's label ends with `?` (on its annotation when it has one,
else on the prediction), the badge adds "flagged", and the status bar counts
flags next to the TP/FP/FN counts. With an annotation selected, `c` comments on that annotation rather
than the focused item, so a box drawn around a questionable miss can be flagged
while the prediction beside it stays in focus. Pressing `c` on a flagged item shows who
flagged it and when, and who last edited the comment, with Save flag to edit the
comment and Resolve flag to close it; a resolved flag stays in
`review_stats.json` with who resolved it and when. Review status "Flagged"
narrows the queue to open flags, and Image status "Flagged" narrows the image
list to images holding any. Flags are not part of undo.

With no review queue (a blind image, or one without predictions), `c` flags the
selected annotation instead. That flag stays on the annotation once predictions
are shown, even when it becomes a TP.

Rejecting an item deletes its annotation, and right-click deletes one directly;
either way an open flag on that annotation has nothing left to open from, so it
is resolved with `resolved_note` set to `rejected` or `deleted`. Undoing brings
the annotation back but leaves the flag resolved. Rejecting a flagged TP keeps
the flag open, since its prediction stays as an FP.

### Completion

Completed is the one dataset gate: ticking it writes a completion record and
saves; unticking it removes the record. If queue items are still missing a
verdict, the pending count shows alongside the TP/FP/FN counts in the status
bar (e.g. "TP 1  FP 1  FN 0  (2 not reviewed)"), but the tick is never
blocked on it.

Stepping to the next image (`Right` or Next) from an image not marked Completed
opens a prompt with three buttons: Mark completed and continue (`Enter`),
Continue without marking (`Right`, so a second press steps on), and Stay (`Esc`).
Going to the previous image or jumping from the image list never asks.

### What is written where

`state/review_stats.json`: `settings.conf_threshold` and `settings.conf_for_import`
(the manifest `imported_at` it was typed against); per image, a `verdicts`
dict keyed by prediction id (a model miss is keyed by its annotation's id
instead), each verdict recording `action` (`accepted` / `rejected`),
`kind` (`fp` / `fn` / `tp`), `class_id`, `conf`, `iou`, `by` and `at`;
per image, a `flags` dict under the same keys (or an annotation's id, for one
flagged with no review queue), each a list of flag entries oldest first,
recording `comment`, `kind` (`null` with no review queue), `class_id`, `by`, `at`,
`edited_by`, `edited_at`, `resolved`, `resolved_by`, `resolved_at` and
`resolved_note`;
a `labels_backed_up` flag set once the first `.original/` backup is made.

`state/annotation_stats.json`: a `completion` entry per completed image,
`{"by", "at", "blind", "annotation_count", "model", "open_flags"}`, where `model`
is the name recorded in the predictions manifest when predictions were visible,
or `null` for a blind pass, and `open_flags` counts the image's items still
flagged when it was marked Completed.

---

## Authors

Zack Loken

---

## License

[MIT](LICENSE)
