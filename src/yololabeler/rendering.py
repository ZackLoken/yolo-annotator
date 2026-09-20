"""Shared canvas-rendering helpers used by both Annotate and Review tabs."""

import tkinter
import tkinter.font as tkFont

_LABEL_NUDGE = 14   # px; roughly one line height at the smallest label size
_LABEL_MAX_NUDGES = 6
_font_cache = {}

_HALO_OFFSETS = [
    (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
    (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
]


PANEL_CORNER_RADIUS = 6   # px; CustomTkinter's default corner_radius, so canvas panels match the toolbar buttons


def rounded_rect(canvas, x0, y0, x1, y1, radius=PANEL_CORNER_RADIUS, **kw):
    """A rectangle with rounded corners, as one smoothed canvas polygon; kw as for create_polygon.

    Each side keeps two collinear control points so the spline stays straight
    between corners; the radius is clamped so short sides do not fold over.
    """
    r = max(0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    points = [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
        x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]
    return canvas.create_polygon(*points, smooth=True, splinesteps=12, **kw)


def halo_text(canvas, x, y, text, fill, **kw):
    """Draw text with a dark halo/shadow for readability on any background."""
    for dx, dy in _HALO_OFFSETS:
        canvas.create_text(x + dx, y + dy, text=text, fill="black", **kw)
    canvas.create_text(x, y, text=text, fill=fill, **kw)


def _cached_font(font):
    """A tkFont.Font for a (family, size[, weight]) tuple, built once per distinct tuple.

    A cached Font is bound to the Tk interpreter live when it was built; if that
    interpreter has since been torn down (each test's own tk.Tk(), for instance,
    while this cache is module-scoped and outlives any one of them), rebuild
    rather than raise "application has been destroyed".
    """
    key = tuple(font)
    fnt = _font_cache.get(key)
    if fnt is not None:
        try:
            fnt.metrics("linespace")
            return fnt
        except tkinter.TclError:
            pass
    fnt = tkFont.Font(
        family=font[0], size=font[1],
        weight=font[2] if len(font) > 2 else "normal")
    _font_cache[key] = fnt
    return fnt


def _overlaps(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def place_label(canvas, placed, x, y, text, fill, anchor="sw", font=None, **kw):
    """halo_text, nudged down until it clears every box already placed this render pass.

    placed is a list of (x0, y0, x1, y1) boxes drawn so far; the resolved box is
    appended to it. anchor is "sw" or "nw", matching the four callers here.
    """
    fnt = _cached_font(font)
    w, h = fnt.measure(text), fnt.metrics("linespace")

    def box_at(yy):
        y0 = (yy - h) if anchor == "sw" else yy
        return (x, y0, x + w, y0 + h)

    box = box_at(y)
    for _ in range(_LABEL_MAX_NUDGES):
        if not any(_overlaps(box, other) for other in placed):
            break
        y += _LABEL_NUDGE
        box = box_at(y)
    placed.append(box)
    halo_text(canvas, x, y, text, fill, anchor=anchor, font=font, **kw)
