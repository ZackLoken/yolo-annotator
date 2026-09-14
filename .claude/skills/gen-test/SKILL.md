---
name: gen-test
description: Use when adding pytest coverage for a yololabeler module that has no matching tests/test_<module>.py, or extending an existing one, so new tests match this repo's conventions.
---

# gen-test

## Overview

This repo's pytest suite (`tests/`) currently covers `annotation/engine.py`, `label_io.py`,
`matching.py`, and `review/engine.py`. `utils.py`, `rendering.py`, and `gui.py` have no
test file yet. Follow the conventions already established in the existing tests rather
than introducing a new style or new test dependencies.

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
| `utils.py` | `auto_orient_image` is pure PIL logic, test directly. | `_load_custom_fonts`, `_get_font_family`, `suppress_tk_mac_warnings` touch font files and Tk internals, low-value and brittle to unit test; skip unless specifically asked. |
| `rendering.py` | `halo_text(canvas, x, y, text, fill, **kw)` needs a real Tkinter `Canvas`, which needs a display. | Don't fabricate a mock `Canvas` that never exercises real drawing calls. If no display is available in the run environment, say so instead of writing a test that can't actually run. |
| `gui.py` | `YoloLabeler` (~1900 lines) is a monolithic Tkinter class wired to widgets and file I/O. | Don't attempt full-class unit tests via mocking. Extract and test any pure helper logic if present; otherwise flag GUI coverage as a manual-testing concern rather than writing brittle mocks. |

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
