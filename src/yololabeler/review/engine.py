"""ReviewEngine — Review logic, detection matching, accept/reject.

GUI-free.  Operates on an AppState instance.  Can be instantiated
headlessly for programmatic use (AI agents, training pipelines, CLI).
"""

import datetime
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

from yololabeler.annotation.document import new_annotation
from yololabeler.label_io import write_json_atomic
from yololabeler.matching import compute_matches
from yololabeler.state import AppState
from yololabeler.state_io import read_json_or_quarantine

# Centre-match tolerance carried over from the original code, used by the migration.
MATCH_TOLERANCE = 0.002
# Default prediction confidence cutoff for the document-based queue (spec 4.3);
# supersedes the old tab-level REVIEW_CONF_THRESHOLD constant.
DEFAULT_CONF_THRESHOLD = 0.50


@dataclass(frozen=True)
class QueueItem:
    """One thing to judge: an unmatched prediction, a model miss, or a match."""
    kind: str
    prediction: object
    annotation: object
    iou: Optional[float]

    @property
    def key(self):
        """The prediction id for fp/tp, the annotation id for fn."""
        return self.prediction.id if self.kind in ("fp", "tp") else self.annotation.id

    @property
    def class_id(self):
        """The class id, from the prediction when present, else the annotation."""
        return self.prediction.class_id if self.prediction else self.annotation.class_id


def _split_predictions(predictions):
    """Split a Prediction list into (box list, polygon list), original order kept."""
    boxes = [p for p in predictions if p.kind == "box"]
    polys = [p for p in predictions if p.kind == "polygon"]
    return boxes, polys


def match_document(document, predictions, iou_threshold, conf_threshold):
    """Run compute_matches over a Document and a Prediction list."""
    pboxes, ppolys = _split_predictions(predictions)
    gt_boxes = [(*a.points[0], *a.points[1], a.class_id) for a in document.boxes()]
    gt_polys = [(list(a.points), a.class_id) for a in document.polygons()]
    pred_boxes = [(*p.points[0], *p.points[1], p.class_id, p.confidence) for p in pboxes]
    pred_polys = [(list(p.points), p.class_id, p.confidence) for p in ppolys]
    return compute_matches(gt_boxes, gt_polys, pred_boxes, pred_polys,
                           iou_threshold, conf_threshold)


def build_queue(document, predictions, matches, verdicts, filter_type="all",
                filter_class="all", filter_status="all"):
    """Flatten matches into QueueItems ordered fp, fn, tp, then filter (spec 4.4)."""
    boxes, polys = document.boxes(), document.polygons()
    pboxes, ppolys = _split_predictions(predictions)

    def gt(gt_type, idx):
        return (boxes if gt_type == "box" else polys)[idx]

    def pr(p_type, idx):
        return (pboxes if p_type == "box" else ppolys)[idx]

    items: List[QueueItem] = []
    for p_type, p_idx, _cid, _conf in matches["fp"]:
        items.append(QueueItem("fp", pr(p_type, p_idx), None, None))
    for gt_type, gt_idx, _cid in matches["fn"]:
        items.append(QueueItem("fn", None, gt(gt_type, gt_idx), None))
    for gt_type, gt_idx, p_type, p_idx, iou, _cid, _conf in matches["tp"]:
        items.append(QueueItem("tp", pr(p_type, p_idx), gt(gt_type, gt_idx), iou))

    def keep(item):
        if filter_type != "all" and item.kind != filter_type:
            return False
        if filter_class != "all" and item.class_id != filter_class:
            return False
        reviewed = item.key in verdicts
        if filter_status == "reviewed":
            return reviewed
        if filter_status == "not_reviewed":
            return not reviewed
        return True

    return [item for item in items if keep(item)]


def apply_accept(document, item, user):
    """Accept the item; an fp becomes an annotation (spec 3.3)."""
    if item.kind == "fp":
        p = item.prediction
        created = new_annotation(p.kind, p.points, p.class_id, user, source="accepted",
                                 prediction_id=p.id, confidence=p.confidence)
        document.add(created)
        return "accepted", created
    return ("confirmed" if item.kind == "tp" else "kept"), None


def apply_reject(document, item):
    """Reject the item; a tp or fn loses its annotation (spec 3.3)."""
    if item.annotation is not None:
        return "rejected", document.remove(item.annotation.id)
    return "rejected", None


class ReviewEngine:
    """Review operations that operate on AppState without any GUI dependency.

    on_error is an optional callback taking one message, used to report a
    failed write to whatever feedback channel the caller has; headless
    callers that pass nothing get silence.
    """

    def __init__(self, state: AppState, on_error=None):
        self.state = state
        self.on_error = on_error or (lambda message: None)

    # ── Review state persistence ──────────────────────────────────────────

    def review_state_path(self):
        s = self.state
        if s.image_folder:
            return os.path.join(s.state_dir, "review_stats.json")
        return None

    def load_review_state(self):
        """Read review_stats.json; returns the quarantined path if it was corrupt."""
        path = self.review_state_path()
        moved = None
        data = None
        if path:
            data, moved = read_json_or_quarantine(path)
        self.state._review_state = data or {}
        return moved

    def save_review_state(self):
        """Write review_stats.json atomically."""
        path = self.review_state_path()
        if not path:
            return
        try:
            write_json_atomic(path, self.state._review_state)
        except OSError as e:
            self.on_error(f"Could not save review_stats.json: {e.strerror or e}")

    # ── Verdicts (document-based queue) ────────────────────────────────────

    @property
    def conf_threshold(self):
        """The persisted prediction confidence cutoff, DEFAULT_CONF_THRESHOLD if unset."""
        settings = self.state._review_state.get("settings", {})
        return float(settings.get("conf_threshold", DEFAULT_CONF_THRESHOLD))

    @conf_threshold.setter
    def conf_threshold(self, value):
        self.state._review_state.setdefault("settings", {})["conf_threshold"] = float(value)
        self.save_review_state()

    def _image_entry(self, img_name):
        per_image = self.state._review_state.setdefault("image", {})
        return per_image.setdefault(img_name, {"img_status": "not_started", "detections": []})

    def verdicts(self, img_name):
        """The live verdict dict for an image, keyed by QueueItem.key."""
        return self._image_entry(img_name).setdefault("verdicts", {})

    def record_verdict(self, img_name, item, action, user):
        """Record the outcome of a queue item review and save."""
        self.verdicts(img_name)[item.key] = {
            "action": action, "kind": item.kind, "class_id": item.class_id,
            "conf": item.prediction.confidence if item.prediction else None,
            "iou": round(item.iou, 4) if item.iou is not None else None,
            "by": user, "at": datetime.datetime.now().isoformat(timespec="seconds")}
        self.save_review_state()

    def remove_verdict(self, img_name, key):
        """Remove a recorded verdict, if present, and save."""
        self.verdicts(img_name).pop(key, None)
        self.save_review_state()

    def migrate_centre_entries(self, img_name, predictions, width, height):
        """Re-key old centre-matched verdict entries by prediction id (spec 6.3)."""
        entry = self.state._review_state.get("image", {}).get(img_name)
        if not entry or "detections" not in entry:
            return 0
        legacy = entry.pop("detections")
        verdicts = self.verdicts(img_name)
        dropped = 0
        for old in legacy:
            centre = old.get("pred_bbox_norm")
            if not centre:
                dropped += 1
                continue
            cx, cy = centre[0] * width, centre[1] * height
            hit = None
            for p in predictions:
                xs = [pt[0] for pt in p.points]
                ys = [pt[1] for pt in p.points]
                pcx, pcy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                if (abs(pcx - cx) / width < MATCH_TOLERANCE
                        and abs(pcy - cy) / height < MATCH_TOLERANCE):
                    hit = p
                    break
            if hit is None:
                dropped += 1
                continue
            verdicts[hit.id] = {
                "action": old.get("action", "reviewed"),
                "kind": old.get("match_type", "").lower(),
                "class_id": old.get("class_id"), "conf": old.get("conf"),
                "iou": old.get("iou"), "by": old.get("reviewed_by", ""), "at": ""}
        self.save_review_state()
        return dropped

    # ── Label backup ──────────────────────────────────────────────────────

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

