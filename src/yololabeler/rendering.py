"""Shared canvas-rendering helpers used by both Annotate and Review tabs."""

_HALO_OFFSETS = [
    (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
    (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
]


def halo_text(canvas, x, y, text, fill, **kw):
    """Draw text with a dark halo/shadow for readability on any background."""
    for dx, dy in _HALO_OFFSETS:
        canvas.create_text(x + dx, y + dy, text=text, fill="black", **kw)
    canvas.create_text(x, y, text=text, fill=fill, **kw)
