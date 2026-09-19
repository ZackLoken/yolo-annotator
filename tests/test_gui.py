"""Tests for yololabeler.gui, limited to what needs no Tk display.

YoloLabeler forwards the attribute names in _STATE_ATTRS to its AppState.
A name present on AppState but missing from _STATE_ATTRS silently splits
into two attributes (one on the window, one on the state), so the two sets
must match exactly.
"""

import pytest

from yololabeler.state import AppState

gui = pytest.importorskip("yololabeler.gui")


# ── _STATE_ATTRS ────────────────────────────────────────────────────────────

class TestStateAttrs:
    def test_every_appstate_attribute_is_forwarded(self):
        missing = set(vars(AppState())) - gui.YoloLabeler._STATE_ATTRS
        assert missing == set()

    def test_every_forwarded_name_exists_on_appstate(self):
        stale = gui.YoloLabeler._STATE_ATTRS - set(vars(AppState()))
        assert stale == set()


# ── window_geometry ─────────────────────────────────────────────────────────

class TestWindowGeometry:
    def parse(self, spec):
        size, x, y = spec.split("+")
        w, h = size.split("x")
        return int(w), int(h), int(x), int(y)

    def test_a_roomy_screen_gets_the_asked_for_size_centred(self):
        w, h, x, y = self.parse(gui.window_geometry(2560, 1440, 1600, 800))
        assert (w, h) == (1600, 800)
        assert x == (2560 - 1600) // 2

    def test_a_narrow_screen_clamps_the_window_inside_it(self):
        w, h, x, y = self.parse(gui.window_geometry(1366, 768, 1600, 800))
        assert w <= 1366 and h <= 768
        assert x + w <= 1366 and y + h <= 768

    def test_the_window_never_starts_off_the_left_or_top(self):
        for screen in ((1024, 600), (1366, 768), (3840, 2160)):
            _, _, x, y = self.parse(gui.window_geometry(*screen, 1600, 800))
            assert x >= 0 and y >= 0

    def test_the_taskbar_allowance_keeps_the_bottom_clear(self):
        w, h, x, y = self.parse(gui.window_geometry(1920, 1080, 1600, 1080))
        assert y + h < 1080
