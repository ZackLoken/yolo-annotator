"""ReviewEngine — Review logic, detection matching, accept/reject.

GUI-free.  Operates on an AppState instance.  Can be instantiated
headlessly for programmatic use (AI agents, training pipelines, CLI).
"""

import json
import os
import shutil

from yololabeler.state import AppState
from yololabeler.label_io import write_detect_labels, write_segment_labels


class ReviewEngine:
    """Review operations that operate on AppState without any GUI dependency."""

    def __init__(self, state: AppState):
        self.state = state

    # ── Review state persistence ──────────────────────────────────────────

    def review_state_path(self):
        s = self.state
        if s.image_folder:
            return os.path.join(s.state_dir, "review_stats.json")
        return None

    def load_review_state(self):
        path = self.review_state_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.state._review_state = json.load(f)
            except Exception:
                self.state._review_state = {}
        else:
            self.state._review_state = {}
        self.invalidate_reviewed_lookup()

    def save_review_state(self):
        path = self.review_state_path()
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state._review_state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save review state: {e}")

    # ── Image review status ───────────────────────────────────────────────

    def mark_image_reviewed(self, img_name):
        """Mark an image as fully reviewed in review_stats.json."""
        per_image = self.state._review_state.setdefault("image", {})
        img_data = per_image.setdefault(
            img_name, {"img_status": "completed", "detections": []})
        img_data["img_status"] = "completed"
        self.save_review_state()

    def is_image_reviewed(self, img_name):
        """Check if image has been reviewed."""
        per_image = self.state._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return False
        return img_data.get("img_status") == "completed"

    def get_image_review_status(self, img_name):
        """Get review status: 'completed', 'started', or 'not_started'."""
        per_image = self.state._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return "not_started"
        return img_data.get("img_status", "not_started")

    # ── Detection bounding boxes ──────────────────────────────────────────

    def match_bbox(self, gt_type, gt_idx, p_type, p_idx):
        """Compute bounding box for a detection in image coordinates."""
        s = self.state
        bboxes = []
        if gt_type and gt_idx is not None:
            if gt_type == 'box' and 0 <= gt_idx < len(s._review_gt_boxes):
                b = s._review_gt_boxes[gt_idx]
                bboxes.append((b[0], b[1], b[2], b[3]))
            elif (gt_type == 'polygon'
                  and 0 <= gt_idx < len(s._review_gt_polygons)):
                pts = s._review_gt_polygons[gt_idx][0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        if p_type and p_idx is not None:
            if p_type == 'box' and 0 <= p_idx < len(s._review_pred_boxes):
                b = s._review_pred_boxes[p_idx]
                bboxes.append((b[0], b[1], b[2], b[3]))
            elif (p_type == 'polygon'
                  and 0 <= p_idx < len(s._review_pred_polygons)):
                pts = s._review_pred_polygons[p_idx][0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        if not bboxes:
            return (0, 0, s._review_img_w, s._review_img_h)
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return (x1, y1, x2, y2)

    def det_norm_bbox(self, det, which='auto'):
        """Get normalized [cx, cy, w, h] for a detection's GT or pred bbox.

        which: 'gt', 'pred', or 'auto' (pred if available, else gt).
        """
        s = self.state
        img_w = max(s._review_img_w, 1)
        img_h = max(s._review_img_h, 1)

        def _norm(gt_or_pred_type, gt_or_pred_idx, boxes, polygons):
            if gt_or_pred_type == 'box' and gt_or_pred_idx is not None:
                if 0 <= gt_or_pred_idx < len(boxes):
                    b = boxes[gt_or_pred_idx]
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    return [
                        round((x1 + x2) / 2 / img_w, 6),
                        round((y1 + y2) / 2 / img_h, 6),
                        round((x2 - x1) / img_w, 6),
                        round((y2 - y1) / img_h, 6)]
            elif gt_or_pred_type == 'polygon' and gt_or_pred_idx is not None:
                if 0 <= gt_or_pred_idx < len(polygons):
                    pts = (polygons[gt_or_pred_idx][0]
                           if isinstance(polygons[gt_or_pred_idx], tuple)
                           else polygons[gt_or_pred_idx])
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    return [
                        round((x1 + x2) / 2 / img_w, 6),
                        round((y1 + y2) / 2 / img_h, 6),
                        round((x2 - x1) / img_w, 6),
                        round((y2 - y1) / img_h, 6)]
            return None

        gt_bbox = _norm(det.get('gt_type'), det.get('gt_idx'),
                        s._review_gt_boxes, s._review_gt_polygons)
        pred_bbox = _norm(det.get('pred_type'), det.get('pred_idx'),
                          s._review_pred_boxes, s._review_pred_polygons)

        if which == 'gt':
            return gt_bbox
        elif which == 'pred':
            return pred_bbox
        return pred_bbox or gt_bbox  # auto

    # ── Reviewed entry lookup (spatial hash) ──────────────────────────────

    def build_reviewed_lookup(self, img_name):
        """Build O(1) lookup dicts for reviewed entries on an image."""
        s = self.state
        per_image = s._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            s._reviewed_lookup = (img_name, {}, {})
            return
        reviewed_dets = img_data.get("detections", [])
        QUANT = 500
        pred_map = {}
        gt_map = {}
        for entry in reviewed_dets:
            e_pred = entry.get("pred_bbox_norm")
            if e_pred:
                qk = (round(e_pred[0] * QUANT), round(e_pred[1] * QUANT))
                pred_map.setdefault(qk, []).append(entry)
            e_gt = entry.get("gt_bbox_norm")
            if e_gt:
                qk = (round(e_gt[0] * QUANT), round(e_gt[1] * QUANT))
                gt_map.setdefault(qk, []).append(entry)
        s._reviewed_lookup = (img_name, pred_map, gt_map)

    def invalidate_reviewed_lookup(self):
        """Mark the reviewed-entry lookup as stale."""
        self.state._reviewed_lookup = ("", {}, {})

    def find_reviewed_entry(self, det, img_name):
        """Find a matching reviewed entry for a detection.

        Uses a spatial hash for O(1) amortized lookup.
        """
        s = self.state
        if not img_name:
            return None

        if s._reviewed_lookup[0] != img_name:
            self.build_reviewed_lookup(img_name)

        _, pred_map, gt_map = s._reviewed_lookup

        TOLERANCE = 0.002
        QUANT = 500

        det_type = det['det_type']
        if det_type in ('tp', 'fp'):
            pred_bbox = self.det_norm_bbox(det, 'pred')
            if not pred_bbox:
                return None
            pcx, pcy = pred_bbox[0], pred_bbox[1]
            qx, qy = round(pcx * QUANT), round(pcy * QUANT)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for entry in pred_map.get((qx + dx, qy + dy), ()):
                        e_pred = entry.get("pred_bbox_norm")
                        if (e_pred
                                and abs(e_pred[0] - pcx) < TOLERANCE
                                and abs(e_pred[1] - pcy) < TOLERANCE):
                            return entry
        else:  # fn
            gt_bbox = self.det_norm_bbox(det, 'gt')
            if not gt_bbox:
                return None
            gcx, gcy = gt_bbox[0], gt_bbox[1]
            qx, qy = round(gcx * QUANT), round(gcy * QUANT)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for entry in gt_map.get((qx + dx, qy + dy), ()):
                        e_gt = entry.get("gt_bbox_norm")
                        if (e_gt
                                and abs(e_gt[0] - gcx) < TOLERANCE
                                and abs(e_gt[1] - gcy) < TOLERANCE):
                            return entry
        return None

    # ── Detection list (filtered) ─────────────────────────────────────────

    def rebuild_review_detections(self):
        """Build the filtered detection list from current matches."""
        s = self.state
        dets = []
        matches = s._review_matches
        if not matches:
            s._review_detections = dets
            return

        ftype = s._review_filter_type
        fclass = s._review_filter_class
        fstatus = s._review_status_filter

        img_name = s.images[s._review_index] if s.images else ""

        def _class_ok(cid):
            if fclass == "all":
                return True
            return cid == fclass

        def _status_ok(det):
            if fstatus == "all":
                return True
            entry = self.find_reviewed_entry(det, img_name)
            if fstatus == "reviewed":
                return entry is not None
            return entry is None  # not_reviewed

        if ftype in ("all", "tp"):
            for gt_type, gt_idx, p_type, p_idx, iou, cid, conf in matches['tp']:
                if not _class_ok(cid):
                    continue
                bbox = self.match_bbox(gt_type, gt_idx, p_type, p_idx)
                det = {
                    'det_type': 'tp', 'class_id': cid, 'conf': conf,
                    'iou': iou,
                    'gt_type': gt_type, 'gt_idx': gt_idx,
                    'pred_type': p_type, 'pred_idx': p_idx,
                    'bbox': bbox,
                }
                if _status_ok(det):
                    dets.append(det)

        if ftype in ("all", "fp"):
            for p_type, p_idx, cid, conf in matches['fp']:
                if not _class_ok(cid):
                    continue
                bbox = self.match_bbox(None, None, p_type, p_idx)
                det = {
                    'det_type': 'fp', 'class_id': cid, 'conf': conf,
                    'iou': None,
                    'gt_type': None, 'gt_idx': None,
                    'pred_type': p_type, 'pred_idx': p_idx,
                    'bbox': bbox,
                }
                if _status_ok(det):
                    dets.append(det)

        if ftype in ("all", "fn"):
            for gt_type, gt_idx, cid in matches['fn']:
                if not _class_ok(cid):
                    continue
                bbox = self.match_bbox(gt_type, gt_idx, None, None)
                det = {
                    'det_type': 'fn', 'class_id': cid, 'conf': None,
                    'iou': None,
                    'gt_type': gt_type, 'gt_idx': gt_idx,
                    'pred_type': None, 'pred_idx': None,
                    'bbox': bbox,
                }
                if _status_ok(det):
                    dets.append(det)

        s._review_detections = dets

    # ── Record & check ────────────────────────────────────────────────────

    def record_detection_action(self, det, action):
        """Record a review action ('accepted', 'rejected', 'edited')."""
        s = self.state
        if not s.images:
            return
        img_name = s.images[s._review_index]

        per_image = s._review_state.setdefault("image", {})
        img_data = per_image.setdefault(
            img_name, {"img_status": "started", "detections": []})

        class_name = s.class_names.get(
            det['class_id'], f"class_{det['class_id']}")
        entry = {
            "match_type": det['det_type'].upper(),
            "det_status": "reviewed",
            "action": action,
            "reviewed_by": s._current_user,
            "class_id": det['class_id'],
            "class_name": class_name,
            "gt_bbox_norm": self.det_norm_bbox(det, 'gt'),
            "pred_bbox_norm": self.det_norm_bbox(det, 'pred'),
            "iou": (round(det['iou'], 4)
                    if det.get('iou') is not None else None),
            "conf": (round(det['conf'], 4)
                     if det.get('conf') is not None else None),
        }

        existing = self.find_reviewed_entry(det, img_name)
        if existing:
            idx = img_data["detections"].index(existing)
            img_data["detections"][idx] = entry
        else:
            img_data["detections"].append(entry)

        self.invalidate_reviewed_lookup()

        if img_data.get("img_status") == "not_started":
            img_data["img_status"] = "started"

        self.save_review_state()

    def check_image_review_complete(self):
        """If all detections reviewed, mark image complete."""
        s = self.state
        if not s.images:
            return
        img_name = s.images[s._review_index]

        matches = s._review_matches
        if not matches:
            return

        total = (len(matches.get('tp', []))
                 + len(matches.get('fp', []))
                 + len(matches.get('fn', [])))
        if total == 0:
            return

        per_image = s._review_state.get("image", {})
        img_data = per_image.get(img_name)
        if not img_data:
            return

        reviewed_count = len(img_data.get("detections", []))
        if reviewed_count >= total:
            img_data["img_status"] = "completed"
            self.save_review_state()

    # ── Label backup & save ───────────────────────────────────────────────

    def backup_original_labels(self):
        """Copy label files to .original/ on first review session."""
        s = self.state
        if s._review_state.get("labels_backed_up"):
            return
        for label_dir in (s.detect_dir, s.segment_dir):
            if not os.path.isdir(label_dir):
                continue
            backup_dir = os.path.join(label_dir, ".original")
            if os.path.isdir(backup_dir):
                continue
            txt_files = [
                f for f in os.listdir(label_dir)
                if f.endswith(".txt")
                and os.path.isfile(os.path.join(label_dir, f))]
            if not txt_files:
                continue
            os.makedirs(backup_dir, exist_ok=True)
            for fname in txt_files:
                src = os.path.join(label_dir, fname)
                dst = os.path.join(backup_dir, fname)
                shutil.copy2(src, dst)
        s._review_state["labels_backed_up"] = True
        self.save_review_state()

    def save_gt(self):
        """Write current review GT boxes/polygons back to label files."""
        s = self.state
        if not s.images:
            return
        img_name = s.images[s._review_index]
        stem = os.path.splitext(img_name)[0]

        detect_path = os.path.join(s.detect_dir, f"{stem}.txt")
        try:
            write_detect_labels(
                detect_path, s._review_gt_boxes,
                s._review_img_w, s._review_img_h)
        except Exception as e:
            print(f"Warning: Could not save detect labels: {e}")

        segment_path = os.path.join(s.segment_dir, f"{stem}.txt")
        try:
            write_segment_labels(
                segment_path, s._review_gt_polygons,
                s._review_img_w, s._review_img_h)
        except Exception as e:
            print(f"Warning: Could not save segment labels: {e}")
