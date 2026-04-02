"""Geometry helpers and GT-vs-prediction matching engine.

All functions are pure (no GUI dependencies).  The ``compute_matches``
function is the main entry point used by the Review tab.
"""

import math
from collections import defaultdict

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid


# ── Geometry primitives ─────────────────────────────────────────────────────

def point_to_segment_dist(px, py, ax, ay, bx, by):
    """Return the distance from point *(px, py)* to segment *AB*."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_in_polygon(px, py, points):
    """Ray-casting test: *True* if *(px, py)* is inside *points*."""
    n = len(points)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def box_iou(b1, b2):
    """Compute IoU between two boxes ``(x1, y1, x2, y2, ...)``."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def polygon_iou(geom1, area1, geom2, area2):
    """Compute IoU between two pre-built Shapely geometries."""
    try:
        inter = geom1.intersection(geom2).area
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union
    except Exception:
        return 0.0


def box_to_points(box):
    """Convert ``(x1, y1, x2, y2, ...)`` to four polygon vertices."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# ── Matching engine ─────────────────────────────────────────────────────────

def compute_matches(gt_boxes, gt_polygons, pred_boxes, pred_polygons,
                    iou_threshold=0.5, conf_threshold=0.25):
    """Match predictions to GT and classify as TP / FP / FN.

    Uses greedy matching: sort all same-class GT-Pred pairs by IoU
    descending, then assign greedily.

    Parameters
    ----------
    gt_boxes : list[tuple]
        ``(x1, y1, x2, y2, class_id)``
    gt_polygons : list[tuple]
        ``(points, class_id)``
    pred_boxes : list[tuple]
        ``(x1, y1, x2, y2, class_id, conf)``
    pred_polygons : list[tuple]
        ``(points, class_id, conf)``
    iou_threshold : float
    conf_threshold : float

    Returns
    -------
    dict with keys:
      ``'tp'``: ``[(gt_type, gt_idx, pred_type, pred_idx, iou, class_id, conf), ...]``
      ``'fp'``: ``[(pred_type, pred_idx, class_id, conf), ...]``
      ``'fn'``: ``[(gt_type, gt_idx, class_id), ...]``
    """
    # Build unified lists
    gt_items = []
    for i, (x1, y1, x2, y2, cid) in enumerate(gt_boxes):
        gt_items.append(('box', i, cid, (x1, y1, x2, y2)))
    for i, (pts, cid) in enumerate(gt_polygons):
        gt_items.append(('polygon', i, cid, pts))

    pred_items = []
    for i, (x1, y1, x2, y2, cid, conf) in enumerate(pred_boxes):
        if conf < conf_threshold:
            continue
        pred_items.append(('box', i, cid, conf, (x1, y1, x2, y2)))
    for i, (pts, cid, conf) in enumerate(pred_polygons):
        if conf < conf_threshold:
            continue
        pred_items.append(('polygon', i, cid, conf, pts))

    # Group by class
    gt_by_class = defaultdict(list)
    for gi, item in enumerate(gt_items):
        gt_by_class[item[2]].append(gi)
    pred_by_class = defaultdict(list)
    for pi, item in enumerate(pred_items):
        pred_by_class[item[2]].append(pi)

    # Lazy Shapely geometry caches
    gt_geom_cache = {}
    pred_geom_cache = {}

    def _get_gt_geom(gi):
        if gi not in gt_geom_cache:
            gt_type, _, _, data = gt_items[gi]
            pts = box_to_points(data) if gt_type == 'box' else data
            g = ShapelyPolygon(pts)
            if not g.is_valid:
                g = make_valid(g)
            gt_geom_cache[gi] = (g, g.area)
        return gt_geom_cache[gi]

    def _get_pred_geom(pi):
        if pi not in pred_geom_cache:
            p_type, _, _, _, data = pred_items[pi]
            pts = box_to_points(data) if p_type == 'box' else data
            g = ShapelyPolygon(pts)
            if not g.is_valid:
                g = make_valid(g)
            pred_geom_cache[pi] = (g, g.area)
        return pred_geom_cache[pi]

    # Compute IoU pairs, grouped by class
    pairs = []
    for cid in gt_by_class:
        if cid not in pred_by_class:
            continue
        gt_indices = gt_by_class[cid]
        pred_indices = pred_by_class[cid]
        for gi in gt_indices:
            gt_type = gt_items[gi][0]
            gt_data = gt_items[gi][3]
            for pi in pred_indices:
                p_type = pred_items[pi][0]
                p_data = pred_items[pi][4]
                if gt_type == 'box' and p_type == 'box':
                    iou = box_iou(gt_data, p_data)
                else:
                    g1, a1 = _get_gt_geom(gi)
                    g2, a2 = _get_pred_geom(pi)
                    iou = polygon_iou(g1, a1, g2, a2)
                if iou >= iou_threshold:
                    pairs.append((iou, gi, pi))

    # Greedy matching (descending IoU)
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_gt = set()
    matched_pred = set()
    tp_list = []

    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        gt_type, gt_idx, gt_cid, _ = gt_items[gi]
        p_type, p_idx, p_cid, p_conf, _ = pred_items[pi]
        tp_list.append((gt_type, gt_idx, p_type, p_idx, iou, gt_cid, p_conf))

    # Unmatched predictions → FP
    fp_list = []
    for pi, (p_type, p_idx, p_cid, p_conf, _) in enumerate(pred_items):
        if pi not in matched_pred:
            fp_list.append((p_type, p_idx, p_cid, p_conf))

    # Unmatched GT → FN
    fn_list = []
    for gi, (gt_type, gt_idx, gt_cid, _) in enumerate(gt_items):
        if gi not in matched_gt:
            fn_list.append((gt_type, gt_idx, gt_cid))

    return {'tp': tp_list, 'fp': fp_list, 'fn': fn_list}
