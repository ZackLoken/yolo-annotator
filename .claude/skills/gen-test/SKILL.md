---
name: gen-test
description: Use when adding pytest coverage for a yololabeler module that has no matching tests/test_<module>.py, or extending an existing one, so new tests match this repo's conventions.
---

# gen-test

## Overview

This repo's pytest suite (`tests/`) has a test file for nearly every module; see the
table below for what each one can and can't cover. `review/panel.py` is the one module
with no dedicated file, its widget-building and stepping logic is exercised through the
headed smoke test in `tests/test_smoke_gui.py` instead. Follow the conventions already
established in the existing tests rather than introducing a new style or new test
dependencies.

## Conventions used in this repo

- One test file per module: `tests/test_<module>.py`.
- Module docstring naming the module under test, e.g. `"""Tests for yololabeler.matching, geometry helpers and matching engine."""`.
- One `TestXxx` class per function/behavior, grouped under a `# ── name ── ...` comment banner.
- Plain `assert`; `pytest.approx(...)` for float comparisons.
- No mocking framework in use (`pyproject.toml` lists only Pillow, customtkinter, shapely as
  dependencies, no `pytest-mock`, no `unittest.mock` elsewhere in the suite). Don't add one
  without checking with the user first.
- Imports: `from yololabeler.<module> import <names>`.

## What's realistically testable here

| Module | Testability | Notes |
|---|---|---|
| `annotation/document.py` | `Annotation` and `Document` are plain dataclasses over tuples and dicts; `load_document`/`save_document` only touch the filesystem via `tmp_path`. Fully testable headless. | No display, no widgets; test round trips through label lines and the sidecar directly. |
| `state_io.py` | `AnnotationStats` and `read_json_or_quarantine` are pure JSON file I/O. Fully testable headless. | Use `tmp_path` for the corrupt-file quarantine case; assert the file is renamed, not deleted. |
| `predictions/store.py` | `Prediction`, `file_hash`, `prediction_id`, `load_predictions`, manifest read/write are pure I/O and hashing over real files in `tmp_path`. Fully testable headless. | No display needed; `file_hash` needs real bytes on disk, not a mock. |
| `predictions/importers.py` | The three converters (`FORMATS`) and `import_predictions` are pure transforms over files in `tmp_path`, including a real `PIL.Image` for EXIF orientation. Fully testable headless. | Cover each format's refusal/rejection case (missing class id, a line with no confidence column, a source file with no matching image) alongside the happy path. |
| `keybindings.py` | `KEY_BINDINGS`, `MOUSE_HELP`, `help_lines`, `readme_tables` are plain data and string building. Fully testable headless. | Assert every binding's label appears in `help_lines` output when its `when` applies, and that the README's `<!-- controls:start -->` block matches `readme_tables()` exactly. |
| `utils.py` | `auto_orient_image` is pure PIL logic, test directly. | `_load_custom_fonts`, `_get_font_family`, `suppress_tk_mac_warnings` touch font files and Tk internals, low-value and brittle to unit test; skip unless specifically asked. |
| `rendering.py` | `halo_text(canvas, x, y, text, fill, **kw)` needs a real Tkinter `Canvas`, which needs a display. | Don't fabricate a mock `Canvas` that never exercises real drawing calls. If no display is available in the run environment, say so instead of writing a test that can't actually run. |
| `review/layer.py` | `draw_prediction_layer` draws onto a real Tkinter `Canvas` the same way `rendering.py` does; needs a display. | Skip like `test_rendering.py` when no display is available. Assert on canvas item counts, tags, colors and text rather than pixels. |
| `review/panel.py` | `ReviewPanel` builds live CustomTkinter widgets in its constructor; there is no headless way to instantiate one. | No dedicated `tests/test_review_panel.py`; its behavior (stepping, focus, filters, threshold, accept/reject) is covered through the headed `app` fixture in `tests/test_smoke_gui.py`. Don't try to build one with a mock parent frame. |
| `gui.py` | `YoloLabeler` is a monolithic Tkinter class wired to widgets and file I/O; covered by headed smoke tests only. | `tests/test_gui.py` covers the one headless-safe invariant, `_STATE_ATTRS` matching `AppState`. Everything else that needs a live window is in `tests/test_smoke_gui.py`. Don't attempt full-class unit tests via mocking. |

## Example (matches `tests/test_matching.py` style)

```python
"""Tests for yololabeler.utils, pure helper functions."""

import pytest
from PIL import Image

from yololabeler.utils import auto_orient_image


# ── auto_orient_image ───────────────────────────────────────────────────────


class TestAutoOrientImage:
    def test_no_exif_returns_same_size(self):
        img = Image.new("RGB", (10, 20))
        result = auto_orient_image(img)
        assert result.size == (10, 20)
```

## Common mistakes

- Inventing a fixture/mocking library this repo doesn't use, check `pyproject.toml`
  dependencies before adding one.
- Writing a test for Tk-dependent code that can't run without a display, instead of
  saying so.
- Naming or grouping that doesn't match the existing files; inconsistent naming
  makes it hard to see what's actually covered.
