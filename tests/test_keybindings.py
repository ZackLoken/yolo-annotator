"""Tests for yololabeler.keybindings, the single source of key help."""

import re
from pathlib import Path

from yololabeler.keybindings import (
    KEY_BINDINGS, MOUSE_HELP, README_MARKERS, help_lines, readme_tables,
)

README = Path(__file__).resolve().parents[1] / "README.md"


# ── table shape ─────────────────────────────────────────────────────────────

class TestTable:
    def test_actions_unique(self):
        actions = [b.action for b in KEY_BINDINGS]
        assert len(actions) == len(set(actions))

    def test_sequences_unique(self):
        seqs = [s for b in KEY_BINDINGS for s in b.sequences]
        assert len(seqs) == len(set(seqs))

    def test_required_actions_present(self):
        actions = {b.action for b in KEY_BINDINGS}
        assert {"prev_image", "next_image", "prev_item", "next_item", "accept", "reject",
                "edit_pair", "fit", "zoom_item", "toggle_mode", "toggle_snap",
                "toggle_stream", "undo", "redo", "save", "click", "escape", "help"} <= actions
        assert all(f"class_{n}" in actions for n in range(10))


# ── help_lines ──────────────────────────────────────────────────────────────

class TestHelpLines:
    def test_every_binding_label_appears_when_applicable(self):
        text = "\n".join(help_lines("polygon", has_queue=True, has_pair=True))
        for b in KEY_BINDINGS:
            assert b.label in text, b.action
        for label, _, when in MOUSE_HELP:
            assert (label in text) == (when != "box"), label

    def test_polygon_only_rows_hidden_in_box_mode(self):
        text = "\n".join(help_lines("box", has_queue=False, has_pair=False))
        assert "Toggle vertex snapping" not in text
        assert "Accept focused item" not in text
        assert "Toggle box / polygon mode" in text
        assert "Drag vertex" not in text and "Close polygon" not in text
        assert "Drag a corner" in text and "Middle-click drag" in text


# ── README ──────────────────────────────────────────────────────────────────

class TestReadme:
    def test_readme_controls_are_generated(self):
        text = README.read_text(encoding="utf-8")
        start, end = README_MARKERS
        block = text.split(start)[1].split(end)[0]
        assert block.strip() == readme_tables().strip()
