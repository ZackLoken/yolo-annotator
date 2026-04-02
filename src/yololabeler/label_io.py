"""YOLO label file I/O — parse and write detect / segment label files.

All functions are pure (no GUI dependencies) and operate on file paths
with explicit image dimensions for coordinate conversion.
"""

import os


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_detect_labels(path, img_w, img_h):
    """Parse a YOLO detect label file into pixel-coordinate boxes.

    Format per line: ``class_id cx cy w h`` (normalised 0-1).

    Returns
    -------
    boxes : list[tuple]
        Each element is ``(x1, y1, x2, y2, class_id)`` in pixel coords.
    class_ids : set[int]
        Set of class IDs found (for bookkeeping by the caller).
    """
    boxes = []
    class_ids = set()
    if not os.path.exists(path):
        return boxes, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            xc = vals[0] * img_w
            yc = vals[1] * img_h
            w = vals[2] * img_w
            h = vals[3] * img_h
            boxes.append((xc - w / 2, yc - h / 2,
                          xc + w / 2, yc + h / 2, class_id))
            class_ids.add(class_id)
    return boxes, class_ids


def parse_segment_labels(path, img_w, img_h):
    """Parse a YOLO segment label file into pixel-coordinate polygons.

    Format per line: ``class_id x1 y1 x2 y2 ...`` (normalised 0-1).

    Returns
    -------
    polygons : list[tuple]
        Each element is ``(points, class_id)`` where *points* is a list
        of ``(px, py)`` tuples in pixel coords.
    class_ids : set[int]
        Set of class IDs found.
    """
    polygons = []
    class_ids = set()
    if not os.path.exists(path):
        return polygons, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or len(parts) % 2 != 1:
                continue
            class_id = int(parts[0])
            vals = [float(v) for v in parts[1:]]
            points = []
            for i in range(0, len(vals), 2):
                px = vals[i] * img_w
                py = vals[i + 1] * img_h
                points.append((px, py))
            polygons.append((points, class_id))
            class_ids.add(class_id)
    return polygons, class_ids


def parse_detect_predictions(path, img_w, img_h):
    """Parse a YOLO detect prediction file (with confidence).

    Format per line: ``class_id confidence cx cy w h`` (normalised 0-1).

    Returns
    -------
    pred_boxes : list[tuple]
        Each element is ``(x1, y1, x2, y2, class_id, conf)``.
    class_ids : set[int]
    """
    pred_boxes = []
    class_ids = set()
    if not os.path.exists(path):
        return pred_boxes, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            class_id = int(parts[0])
            conf = float(parts[1])
            vals = [float(v) for v in parts[2:]]
            xc = vals[0] * img_w
            yc = vals[1] * img_h
            w = vals[2] * img_w
            h = vals[3] * img_h
            pred_boxes.append((xc - w / 2, yc - h / 2,
                               xc + w / 2, yc + h / 2,
                               class_id, conf))
            class_ids.add(class_id)
    return pred_boxes, class_ids


def parse_segment_predictions(path, img_w, img_h):
    """Parse a YOLO segment prediction file (with confidence).

    Format per line: ``class_id confidence x1 y1 x2 y2 ...`` (normalised).

    Returns
    -------
    pred_polygons : list[tuple]
        Each element is ``(points, class_id, conf)``.
    class_ids : set[int]
    """
    pred_polygons = []
    class_ids = set()
    if not os.path.exists(path):
        return pred_polygons, class_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            if (len(parts) - 2) % 2 != 0:
                continue
            class_id = int(parts[0])
            conf = float(parts[1])
            vals = [float(v) for v in parts[2:]]
            points = []
            for i in range(0, len(vals), 2):
                px = vals[i] * img_w
                py = vals[i + 1] * img_h
                points.append((px, py))
            pred_polygons.append((points, class_id, conf))
            class_ids.add(class_id)
    return pred_polygons, class_ids


# ── Writing ─────────────────────────────────────────────────────────────────

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
    if boxes:
        with open(path, "w", encoding="utf-8") as f:
            for x1, y1, x2, y2, cls in boxes:
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    elif os.path.exists(path):
        os.remove(path)


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
    if polygons:
        with open(path, "w", encoding="utf-8") as f:
            for points, cls in polygons:
                coords = " ".join(
                    f"{x / img_w:.6f} {y / img_h:.6f}"
                    for x, y in points)
                f.write(f"{cls} {coords}\n")
    elif os.path.exists(path):
        os.remove(path)
