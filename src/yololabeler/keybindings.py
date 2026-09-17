"""Key bindings declared once, used for Tk binding, the help overlay and the README.

Spec 5.3. Adding a key here is the only way to add one; the tests assert the
help text and README stay in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

README_MARKERS = ("<!-- controls:start -->", "<!-- controls:end -->")


@dataclass(frozen=True)
class Binding:
    """One keyboard action: Tk sequences, the label shown to users, and when it applies."""
    action: str
    sequences: Tuple[str, ...]
    label: str
    description: str
    when: str


def _b(action, sequences, label, description, when="always"):
    """Build a Binding, wrapping sequences in a tuple."""
    return Binding(action, tuple(sequences), label, description, when)


KEY_BINDINGS = (
    _b("prev_image", ("<Left>",), "Left", "Previous image"),
    _b("next_image", ("<Right>",), "Right", "Next image"),
    _b("prev_item", ("<Down>",), "Down", "Previous queue item", "queue"),
    _b("next_item", ("<Up>",), "Up", "Next queue item", "queue"),
    _b("accept", ("a",), "a", "Accept focused item (an unmatched prediction becomes a new annotation)", "queue"),
    _b("reject", ("r",), "r", "Reject focused item (deletes its annotation, if any)", "queue"),
    _b("edit_pair", ("e",), "e", "Edit focused item (an unmatched prediction is accepted first)", "queue"),
    _b("comment", ("c",), "c", "Comment on the focused item, flagging it for a second look", "queue"),
    _b("fit", ("f",), "f", "Fit image to window"),
    _b("zoom_item", ("z",), "z", "Zoom to focused item", "queue"),
    _b("toggle_mode", ("m",), "m", "Toggle box / polygon mode"),
    _b("toggle_snap", ("s",), "s", "Toggle vertex snapping", "polygon"),
    _b("toggle_stream", ("v",), "v", "Toggle vertex streaming", "polygon"),
    *[_b(f"class_{n}", (str(n),), str(n), f"Select class {n}") for n in range(10)],
    _b("rename_class", ("<Control-r>", "<Command-r>"), "Ctrl+R", "Rename the active class"),
    _b("undo", ("<Control-z>", "<Command-z>"), "Ctrl+Z", "Undo"),
    _b("redo", ("<Control-y>", "<Command-y>"), "Ctrl+Y", "Redo"),
    _b("save", ("<Control-s>", "<Command-s>"), "Ctrl+S", "Save now"),
    _b("click", ("<space>",), "Space", "Left click at the cursor"),
    _b("escape", ("<Escape>",), "Escape", "Cancel polygon / deselect"),
    _b("help", ("h",), "h", "Toggle this help"),
)

MOUSE_HELP = (
    ("Ctrl+Scroll", "Zoom at cursor", "always"),
    ("Scroll", "Pan up / down", "always"),
    ("Shift+Scroll", "Pan left / right", "always"),
    ("Middle-click drag", "Pan", "always"),
    ("Left-click drag", "Draw a box (anywhere off a box outline, including inside a box)", "box"),
    ("Click a box outline", "Select it", "box"),
    ("Drag a box outline", "Move the whole box", "box"),
    ("Drag a corner", "Resize; the opposite corner stays fixed (selected box)", "box"),
    ("Right-click a box outline", "Delete box", "box"),
    ("Left-click", "Place vertex (anywhere off a polygon outline, including inside a polygon)", "polygon"),
    ("Left-click (Stream on)", "Start / pause laying vertices as the pointer moves", "polygon"),
    ("Double-click", "Close polygon", "polygon"),
    ("Click a polygon outline", "Select it", "polygon"),
    ("Drag vertex", "Move vertex (selected polygon)", "polygon"),
    ("Click vertex", "Start a new polygon on it (selected polygon)", "polygon"),
    ("Click edge", "Insert vertex (selected polygon)", "polygon"),
    ("Right-click a vertex", "Delete vertex (selected polygon)", "polygon"),
    ("Right-click a polygon outline", "Delete polygon", "polygon"),
    ("Click Legend", "Open / close the symbology legend (lower left)", "always"),
)


def _applies(when, mode, has_queue, has_pair):
    """Decide whether a binding or mouse row applies to the current context."""
    if when == "always":
        return True
    if when in ("polygon", "box"):
        return mode == when
    if when == "queue":
        return has_queue
    return has_pair


def help_lines(mode, has_queue, has_pair):
    """Lines for the help overlay, filtered to what applies right now."""
    lines = ["── Keyboard ──"]
    class_rows_done = False
    for b in KEY_BINDINGS:
        if not _applies(b.when, mode, has_queue, has_pair):
            continue
        if b.action.startswith("class_"):
            if class_rows_done:
                continue
            class_rows_done = True
            digits = "".join(str(n) for n in range(10))
            lines.append(f"  {digits:<16}Select class by id")
            continue
        lines.append(f"  {b.label:<16}{b.description}")
    lines.append("")
    lines.append("── Mouse ──")
    for label, description, when in MOUSE_HELP:
        if _applies(when, mode, has_queue, has_pair):
            lines.append(f"  {label:<30}{description}")
    return lines


def readme_tables():
    """Markdown tables for the README, one for keys and one for the mouse."""
    out = ["| Action | Key |", "|---|---|"]
    class_done = False
    for b in KEY_BINDINGS:
        if b.action.startswith("class_"):
            if not class_done:
                out.append("| Select class by id | `0`-`9` |")
                class_done = True
            continue
        out.append(f"| {b.description} | `{b.label}` |")
    out += ["", "| Action | Input | Mode |", "|---|---|---|"]
    for label, description, when in MOUSE_HELP:
        out.append(f"| {description} | {label} | {when} |")
    return "\n".join(out)
