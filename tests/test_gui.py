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
