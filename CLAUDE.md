# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install for development: `pip install -e .`
- Run the GUI: `yololabeler`, or `yololabeler /path/to/images`, or `python -m yololabeler /path/to/images`
- Run all tests: `pytest -q`
- Run one test file: `pytest tests/test_matching.py -q`
- Run one test: `pytest tests/test_matching.py::TestBoxIou::test_perfect_overlap -q`
- No linter or formatter is configured in this repo (no ruff/black/flake8 section in pyproject.toml, no config files for any of them).
- CI (`.github/workflows/tests.yml`) runs `pytest -q` on Python 3.9 and 3.12 on push/PR to main.

## Architecture

Strict separation between GUI-free logic and the Tkinter/CustomTkinter GUI layer.

### Data

`state.py` defines `AppState`, the single source of truth for all annotation and review data. Pure data, no GUI, no I/O, no rendering.

### GUI-free logic

Each module's docstring notes it is safe to use headlessly, e.g. for scripting, AI agents, or training pipelines:

- `annotation/engine.py`: `AnnotationEngine`, annotation CRUD, the polygon-bbox spatial index cache, undo/redo. Operates on an `AppState` instance.
- `review/engine.py`: `ReviewEngine`, review state persistence and accept/reject/edit logic. Operates on an `AppState` instance.
- `matching.py`: geometry helpers (point-to-segment distance, point-in-polygon, box/polygon IoU) and `compute_matches`, the greedy highest-IoU-first ground-truth-vs-prediction matcher used by review.
- `label_io.py`: parse/write YOLO detect and segment label files, converting between pixel coordinates and normalized 0-1 coordinates.
- `rendering.py`: canvas-drawing helpers shared by both tabs (currently just `halo_text`, outlined text for readability on any background).
- `utils.py`: font loading, Tk stderr-warning suppression, EXIF auto-orientation.

### GUI layer

One module per tab:

- `annotation/tab.py`: `AnnotateTab`, canvas construction, bindings, pan/zoom, box/polygon interaction, vertex streaming and snapping, save.
- `review/tab.py`: `ReviewTab`, canvas, detection navigation, filter dropdowns, accept/reject/edit actions, Review-to-Annotate handoff.
- `gui.py`: `YoloLabeler`, the main app class. Composes one `AppState`, one `AnnotationEngine`, one `ReviewEngine`, one `AnnotateTab`, one `ReviewTab`.

### Key pattern: state forwarding

`YoloLabeler` does not hold annotation/review data directly. Its `__getattr__`/`__setattr__` (in `gui.py`) transparently forward any attribute name listed in the `_STATE_ATTRS` frozenset to the `AppState` instance at `self._state`. Tab and engine code reads/writes state through `self.app.<attr>` or `self.state.<attr>`, and both land on the same `AppState` object. When adding a new piece of annotation/review state, it must be declared in both `AppState.__init__` and `YoloLabeler._STATE_ATTRS`, or reads/writes through the two paths silently diverge.

### Review matching pipeline

Predictions are loaded from `predictions/detect/` and `predictions/segment/` (same line format as labels, plus a leading confidence column) and matched against ground truth via `matching.compute_matches` using thresholds defined in `review/tab.py` (`REVIEW_IOU_THRESHOLD` 0.60, `REVIEW_CONF_THRESHOLD` 0.50) to classify each as TP/FP/FN, then stepped through one detection at a time in `ReviewTab`.

### On-disk layout

Under the selected image folder: `labels/detect/` and `labels/segment/` for YOLO-format annotations, `state/` for `classes.json`/`annotation_stats.json`/`review_stats.json`, `predictions/detect/` and `predictions/segment/` for model predictions to review. Full label/prediction line formats, the keybinding reference, and the accept/reject/edit behavior matrix are documented in README.md.

## Testing conventions

- One test file per module: `tests/test_<module>.py` (nested modules flatten, e.g. `annotation/engine.py` maps to `tests/test_annotation_engine.py`).
- One `TestXxx` class per function/behavior, grouped under `# --- name --- ...` comment banners (see `tests/test_matching.py` for the exact style).
- Plain `assert`; `pytest.approx(...)` for float comparisons. No mocking framework is used anywhere in the suite.
- `.claude/skills/gen-test/SKILL.md` documents what's realistically testable in each module; large parts of `gui.py` and all of `rendering.py` need a live Tk display. `tests/test_rendering.py` skips itself when no display is available, and `tests/test_gui.py` only checks the `_STATE_ATTRS`/`AppState` invariant, which needs no display.
