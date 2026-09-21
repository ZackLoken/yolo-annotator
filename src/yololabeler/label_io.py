"""YOLO label file I/O: parse and write detect and segment label files.

All functions are pure (no GUI dependencies) and operate on file paths
with explicit image dimensions for coordinate conversion.
"""

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class ParsedRow:
    """One accepted line of a label or prediction file, in pixels."""

    class_id: int
    points: tuple[Point, ...]
    confidence: float | None
    line_index: int
    line: str


@dataclass(frozen=True)
class ParsedFile:
    """Accepted rows plus the 1-based numbers of lines that could not be
    parsed.
    """

    rows: list[ParsedRow]
    rejected: list[int]


def format_detect_line(x1, y1, x2, y2, cls, img_w, img_h):
    """Format one YOLO detect line (no newline) from pixel box corners."""
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def format_segment_line(points, cls, img_w, img_h):
    """Format one YOLO segment line (no newline) from pixel vertices."""
    coords = " ".join(f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in points)
    return f"{cls} {coords}"


def parse_label_file(path, kind, img_w, img_h, with_conf=False):
    """Parse a detect or segment file into pixel rows, reporting rejected
    lines.

    kind is "box" or "polygon"; with_conf expects the confidence in column two,
    the canonical prediction layout.
    """
    rows: list[ParsedRow] = []
    rejected: list[int] = []
    if not os.path.exists(path):
        return ParsedFile(rows, rejected)
    with open(path, encoding="utf-8") as f:
        for index, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            row = _parse_line(line, index, kind, img_w, img_h, with_conf)
            if row is None:
                rejected.append(index + 1)
            else:
                rows.append(row)
    return ParsedFile(rows, rejected)


def _parse_line(line, index, kind, img_w, img_h, with_conf):
    parts = line.split()
    head = 2 if with_conf else 1
    try:
        class_id = int(parts[0])
        confidence = float(parts[1]) if with_conf else None
        vals = [float(v) for v in parts[head:]]
    except ValueError, IndexError:
        return None
    if kind == "box":
        if len(vals) != 4:
            return None
        xc, yc, w, h = (
            vals[0] * img_w,
            vals[1] * img_h,
            vals[2] * img_w,
            vals[3] * img_h,
        )
        points = ((xc - w / 2, yc - h / 2), (xc + w / 2, yc + h / 2))
    else:
        if len(vals) < 6 or len(vals) % 2:
            return None
        points = tuple(
            (vals[i] * img_w, vals[i + 1] * img_h)
            for i in range(0, len(vals), 2)
        )
    return ParsedRow(class_id, points, confidence, index, line)


def write_json_atomic(path, data):
    """Write JSON through a temp file and os.replace so a crash keeps the old
    file.
    """
    path = str(path)
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ── Writing ─────────────────────────────────────────────────────────────────


def _write_label_lines(path, lines):
    """Atomically replace *path* with *lines*, or delete it when there are
    none.

    An empty label file is never left on disk; YOLO treats a missing file
    as an image with no objects.
    """
    if not lines:
        if os.path.exists(path):
            os.remove(path)
        return
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def write_detect_labels(path, boxes, img_w, img_h):
    """Write detect boxes to a YOLO label file.

    Parameters
    ----------
    path : str
        Output file path.
    boxes : list[tuple]
        Each element is ``(x1, y1, x2, y2, class_id)`` in pixel coords.
    img_w, img_h : int
        Image dimensions for normalisation.
    """
    lines = [
        format_detect_line(x1, y1, x2, y2, cls, img_w, img_h) + "\n"
        for x1, y1, x2, y2, cls in boxes
    ]
    _write_label_lines(path, lines)


def write_segment_labels(path, polygons, img_w, img_h):
    """Write segment polygons to a YOLO label file.

    Parameters
    ----------
    path : str
        Output file path.
    polygons : list[tuple]
        Each element is ``(points, class_id)`` where *points* is a list
        of ``(px, py)`` tuples in pixel coords.
    img_w, img_h : int
        Image dimensions for normalisation.
    """
    lines = [
        format_segment_line(points, cls, img_w, img_h) + "\n"
        for points, cls in polygons
    ]
    _write_label_lines(path, lines)
