# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install for development: `pip install -e .`
- Run the GUI: `yololabeler`, or `yololabeler /path/to/images`, or `python -m yololabeler /path/to/images`
- Batch-import predictions headlessly: `yololabeler-import <source_root> <images_root> --format <fmt> --model <name>` (dry run; add `--write` to import)
- Run all tests: `pytest -q`
- Run one test file: `pytest tests/test_matching.py -q`
- Run one test: `pytest tests/test_matching.py::TestBoxIou::test_perfect_overlap -q`
- No linter or formatter is configured in this repo (no ruff/black/flake8 section in pyproject.toml, no config files for any of them).
- CI (`.github/workflows/tests.yml`) runs `pytest -q` on Python 3.9 and 3.12 on push/PR to main.

## Architecture

Strict separation between GUI-free logic and the Tkinter/CustomTkinter GUI layer.

### Data

`state.py` defines `AppState`, the single source of truth for all annotation and review data. Pure data, no GUI, no I/O, no rendering.

### Modules

One canvas: there is no Review tab, no second viewport, and nothing to keep in
sync between them. Each module's docstring notes whether it is safe to use
headlessly, e.g. for scripting, AI agents, or training pipelines.

| Module | Role |
|---|---|
| `annotation/document.py` | `Annotation`, `Document`: label-line formatting, the id-keyed sidecar record, join with hand-edited label lines. GUI-free. |
| `annotation/engine.py` | `AnnotationEngine`: CRUD on the `Document`, the polygon-bbox spatial index cache, undo/redo, save. Operates on an `AppState` instance, GUI-free. |
| `annotation/tab.py` | `AnnotateTab`: the one canvas, construction, bindings, pan/zoom, box/polygon interaction, vertex streaming and snapping, rendering, calls the prediction layer. GUI. |
| `review/engine.py` | `ReviewEngine`: prediction matching against a `Document`, the queue builder, accept/reject, verdict persistence, completion, migrations. Operates on an `AppState` instance, GUI-free. |
| `review/layer.py` | Draws the prediction layer and the focused pair on a canvas passed in by the caller; no widgets of its own. |
| `review/panel.py` | `ReviewPanel`, the status-bar strip: stepping, accept/reject, filters, the confidence entry. Replaces the old `review/tab.py`. GUI. |
| `predictions/store.py` | `Prediction` records, file-hash ids, canonical loading, manifest read/write. GUI-free. |
| `predictions/importers.py` | The three prediction format converters run by Import predictions (`FORMATS`). GUI-free. |
| `predictions/batch.py` | `plan_pairs`, `batch_import`: pair a source prediction tree to an image tree strictly by relative path, then run `import_predictions` per pair. GUI-free. |
| `predictions/cli.py` | `main()` behind the `yololabeler-import` console script: argparse over `batch.py`, dry run by default. GUI-free and headless-safe. |
| `state_io.py` | `AnnotationStats`: `annotation_stats.json` access, quarantine of a corrupt file. GUI-free. |
| `keybindings.py` | `KEY_BINDINGS`, the only place a key is declared; also generates the help overlay and the README Controls tables. GUI-free. |
| `label_io.py` | Parse/write YOLO detect and segment label and prediction files, converting between pixel coordinates and normalized 0-1 coordinates; the shared atomic-write helper. GUI-free. |
| `matching.py` | Geometry helpers (point-to-segment distance, point-in-polygon, box/polygon IoU) and `compute_matches`, the greedy highest-IoU-first matcher. GUI-free. |
| `rendering.py` | `halo_text`, outlined canvas text for readability on any background; used by the annotate canvas and the prediction layer. |
| `utils.py` | Font loading, Tk stderr-warning suppression, EXIF auto-orientation. GUI-free. |
| `gui.py` | `YoloLabeler`, the main app class. Composes one `AppState`, one `AnnotationEngine`, one `ReviewEngine`, one `AnnotateTab`, one `ReviewPanel`; owns the toolbar, `go_to_image`, `save_current`, and the banner. |

`yololabeler/__init__.py` exposes `YoloLabeler` and `main` through a module-level
`__getattr__` rather than importing `gui.py` eagerly, so importing any submodule
(`yololabeler.predictions.cli`, say) does not pull Tk and CustomTkinter in. A new
headless entry point must not undo that by adding a top-level GUI import here.

`go_to_image(index)` in `gui.py` is the only navigation path: it saves the
current image, then loads the requested one. `save_current()` is the only save
path, called by navigation, completion, accept, reject, `Ctrl+S`, and quit.
`KEY_BINDINGS` in `keybindings.py` is the only place a key is bound; adding a
binding there is the only way to add one, and it drives the help overlay and the
README Controls tables. `banner_text`, shown and cleared through
`show_banner`/`clear_banner` in `gui.py`, is the feedback channel while an image
is loaded (import results, save failures, rejected lines all go through it);
`show_canvas_message` covers the same kind of failure when there is no image yet
to draw a banner on (no images in the chosen folder, no loadable images at all).
There are no pop-ups anywhere in the app, and the quit-without-saving
confirmation is the one deliberate modal in the normal workflow.
Verdicts are keyed by prediction id, except a model miss (an annotation with no
matching prediction), which is keyed by the annotation's id instead.

### Key pattern: state forwarding

`YoloLabeler` does not hold annotation/review data directly. Its `__getattr__`/`__setattr__` (in `gui.py`) transparently forward any attribute name listed in the `_STATE_ATTRS` frozenset, including `document`, `verdicts`, `predictions` and `queue`, to the `AppState` instance at `self._state`. Tab and engine code reads/writes state through `self.app.<attr>` or `self.state.<attr>`, and both land on the same `AppState` object. When adding a new piece of annotation/review state, it must be declared in both `AppState.__init__` and `YoloLabeler._STATE_ATTRS`, or reads/writes through the two paths silently diverge.

### On-disk layout

Under the selected image folder: `labels/detect/` and `labels/segment/` for YOLO-format annotations (each with a one-time `.original/` backup before the first destructive edit), `state/` for `classes.json`, `annotation_stats.json`, `review_stats.json`, and one sidecar per image under `state/annotations/<stem>.json`, and `predictions/detect/` and `predictions/segment/` plus `predictions/manifest.json` for imported model predictions. Full label/prediction line formats, the keybinding reference, and the accept/reject verdict table are documented in README.md.

## Testing conventions

- One test file per module: `tests/test_<module>.py` (nested modules flatten, e.g. `annotation/engine.py` maps to `tests/test_annotation_engine.py`).
- One `TestXxx` class per function/behavior, grouped under `# --- name --- ...` comment banners (see `tests/test_matching.py` for the exact style).
- Plain `assert`; `pytest.approx(...)` for float comparisons. No mocking framework is used anywhere in the suite.
- `.claude/skills/gen-test/SKILL.md` documents what's realistically testable in each module; large parts of `gui.py` and all of `rendering.py` need a live Tk display. `tests/test_rendering.py` skips itself when no display is available, and `tests/test_gui.py` only checks the `_STATE_ATTRS`/`AppState` invariant, which needs no display.
