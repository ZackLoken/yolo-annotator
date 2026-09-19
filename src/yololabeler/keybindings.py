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
    _b("prev_item", ("<Down>",), "Down", "Previous item", "queue"),
    _b("next_item", ("<Up>",), "Up", "Next item", "queue"),
    _b("accept", ("a",), "a", "Accept the focused item; an FP becomes an annotation", "queue"),
    _b("reject", ("r",), "r", "Reject the focused item and delete its annotation", "queue"),
    _b("edit_pair", ("e",), "e", "Edit the focused item; an FP is accepted first", "queue"),
    _b("comment", ("c",), "c", "Comment on the focused item and flag it", "queue"),
    _b("fit", ("f",), "f", "Fit the image to the window"),
    _b("zoom_item", ("z",), "z", "Zoom to the focused item", "queue"),
    _b("toggle_mode", ("m",), "m", "Switch between box and polygon mode"),
    _b("toggle_snap", ("s",), "s", "Vertex snapping on or off", "polygon"),
    _b("toggle_stream", ("v",), "v", "Vertex streaming on or off", "polygon"),
    *[_b(f"class_{n}", (str(n),), str(n), f"Pick class {n}") for n in range(10)],
    _b("rename_class", ("<Control-r>", "<Command-r>"), "Ctrl+R", "Rename the active class"),
    _b("undo", ("<Control-z>", "<Command-z>"), "Ctrl+Z", "Undo"),
    _b("redo", ("<Control-y>", "<Command-y>"), "Ctrl+Y", "Redo"),
    _b("save", ("<Control-s>", "<Command-s>"), "Ctrl+S", "Save now"),
    _b("click", ("<space>",), "Space", "Click at the cursor"),
    _b("escape", ("<Escape>",), "Escape", "Cancel the polygon, or deselect"),
    _b("help", ("h",), "h", "Show or hide this help"),
)

MOUSE_HELP = (
    ("Ctrl+Scroll", "Zoom at the cursor", "always"),
    ("Scroll", "Pan up or down", "always"),
    ("Shift+Scroll", "Pan left or right", "always"),
    ("Middle-click drag", "Pan", "always"),
    ("Shift+drag the selected shape", "Move the whole shape", "always"),
    ("Left-click drag", "Draw a box, even inside another", "box"),
    ("Click a box outline", "Select the box", "box"),
    ("Drag a box outline", "Move the box", "box"),
    ("Drag a corner", "Resize the selected box from the opposite corner", "box"),
    ("Right-click a box outline", "Delete the box", "box"),
    ("Left-click", "Place a vertex, even inside another polygon", "polygon"),
    ("Left-click (Stream on)", "Start or pause streaming vertices", "polygon"),
    ("Double-click", "Close the polygon", "polygon"),
    ("Click a polygon outline", "Select the polygon", "polygon"),
    ("Alt+click a polygon", "Select it even where a click would snap", "polygon"),
    ("Drag vertex", "Move a vertex of the selected polygon", "polygon"),
    ("Click vertex", "Start a new polygon on that vertex", "polygon"),
    ("Click edge", "Insert a vertex in the selected polygon", "polygon"),
    ("Right-click a vertex", "Delete a vertex", "polygon"),
    ("Right-click a polygon outline", "Delete the polygon", "polygon"),
    ("Click Legend", "Open or close the legend", "always"),
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
            lines.append(f"  {digits:<16}Pick a class by id")
            continue
        lines.append(f"  {b.label:<16}{b.description}")
    lines.append("")
    lines.append("── Mouse ──")
    for label, description, when in MOUSE_HELP:
        if _applies(when, mode, has_queue, has_pair):
            lines.append(f"  {label:<32}{description}")
    return lines


def readme_tables():
    """Markdown tables for the README, one for keys and one for the mouse."""
    out = ["| Action | Key |", "|---|---|"]
    class_done = False
    for b in KEY_BINDINGS:
        if b.action.startswith("class_"):
            if not class_done:
                out.append("| Pick a class by id | `0`-`9` |")
                class_done = True
            continue
        out.append(f"| {b.description} | `{b.label}` |")
    out += ["", "| Action | Input | Mode |", "|---|---|---|"]
    for label, description, when in MOUSE_HELP:
        out.append(f"| {description} | {label} | {when} |")
    return "\n".join(out)
