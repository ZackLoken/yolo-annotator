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
from yololabeler.predictions.store import read_manifest
from yololabeler.state import AppState
from yololabeler.state_io import read_json_or_quarantine

# Centre-match tolerance carried over from the original code, used by the migration.
MATCH_TOLERANCE = 0.002
# Fallback cutoff when there is no manifest min_conf; the user set it to the conf inference runs at (2026-09-16).
DEFAULT_CONF_THRESHOLD = 0.25


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


def _centre(item):
    """Centre of the item's prediction, or of its annotation when it has no prediction."""
    shape = item.prediction or item.annotation
    xs = [p[0] for p in shape.points]
    ys = [p[1] for p in shape.points]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def spatial_order(items):
    """Order items along a nearest-neighbour path so neighbours are visited in a row.

    The path starts at the item nearest the image's top-left corner and always
    steps to the closest item not yet visited; ties fall back to the item key so
    the order is deterministic.
    """
    remaining = {item.key: item for item in items}
    centres = {key: _centre(item) for key, item in remaining.items()}
    ordered = []
    current = None
    while remaining:
        if current is None:
            key = min(remaining, key=lambda k: (sum(centres[k]), k))
        else:
            cx, cy = centres[current]
            key = min(remaining, key=lambda k: (
                (centres[k][0] - cx) ** 2 + (centres[k][1] - cy) ** 2, k))
        ordered.append(remaining.pop(key))
        current = key
    return ordered


def build_queue(document, predictions, matches, verdicts, filter_type="all",
                filter_class="all", filter_status="all", flagged=frozenset()):
    """Flatten matches into QueueItems in spatial order, then filter (spec 4.4).

    The path is laid over the items the Type and Class filters keep, before the
    Status filter, so judging an item never reorders the ones still to review.
    flagged holds the keys with an open flag, used by the "flagged" status filter.
    """
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

    def keep_type_and_class(item):
        if filter_type != "all" and item.kind != filter_type:
            return False
        return filter_class == "all" or item.class_id == filter_class

    def keep_status(item):
        if filter_status == "flagged":
            return item.key in flagged
        reviewed = item.key in verdicts
        if filter_status == "reviewed":
            return reviewed
        if filter_status == "not_reviewed":
            return not reviewed
        return True

    ordered = spatial_order([item for item in items if keep_type_and_class(item)])
    return [item for item in ordered if keep_status(item)]


def shape_statuses(document, predictions, matches, verdicts):
    """Map each queue item's prediction id and annotation id to its review status.

    The status is the verdict's action, or "not_reviewed" when there is none,
    taken from the unfiltered queue so the active filters never change it. A
    match's prediction and annotation share one status. Returns None when there
    are no matches (no predictions, or a blind image).
    """
    if document is None or not matches:
        return None
    statuses = {}
    for item in build_queue(document, predictions, matches, verdicts):
        verdict = verdicts.get(item.key)
        status = verdict["action"] if verdict else "not_reviewed"
        for shape in (item.prediction, item.annotation):
            if shape is not None:
                statuses[shape.id] = status
    return statuses


def flag_markers(document, predictions, matches, open_keys):
    """Map each open-flagged key still in the queue to the points its marker is drawn at.

    The marker sits on the item's annotation when it has one, else on its
    prediction. Keys whose item no longer exists (a rejected model miss) are left out.
    """
    if document is None or not matches:
        return {}
    return {item.key: (item.annotation or item.prediction).points
            for item in build_queue(document, predictions, matches, {})
            if item.key in open_keys}


def apply_accept(document, item, user):
    """Accept the item; an fp becomes an annotation (spec 3.3)."""
    if item.kind == "fp":
        p = item.prediction
        created = new_annotation(p.kind, p.points, p.class_id, user, source="accepted",
                                 prediction_id=p.id, confidence=p.confidence)
        document.add(created)
        return "accepted", created
    return "accepted", None


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
        """The cutoff typed for the current import, else that import's min_conf, else the default.

        A typed cutoff is stored with the imported_at of the manifest it was typed
        against, so re-running the import resets the cutoff to the new set's own
        minimum instead of keeping a value chosen for older predictions. A cutoff
        saved before this stamp existed carries none and yields to min_conf.
        """
        settings = self.state._review_state.get("settings", {})
        manifest = self._manifest() or {}
        if ("conf_threshold" in settings
                and settings.get("conf_for_import") == manifest.get("imported_at")):
            return float(settings["conf_threshold"])
        min_conf = manifest.get("min_conf")
        return float(min_conf) if min_conf is not None else DEFAULT_CONF_THRESHOLD

    def _manifest(self):
        """predictions/manifest.json for the open folder, or None if absent."""
        if not self.state.image_folder:
            return None
        return read_manifest(os.path.join(self.state.image_folder, "predictions"))

    @conf_threshold.setter
    def conf_threshold(self, value):
        manifest = self._manifest() or {}
        settings = self.state._review_state.setdefault("settings", {})
        settings["conf_threshold"] = float(value)
        settings["conf_for_import"] = manifest.get("imported_at")
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

    def carry_verdict(self, img_name, item, old_verdict):
        """Carry a prior verdict forward onto item's current key after a geometry
        edit reclassified it (e.g. a confirmed match became a standalone miss).
        Keeps the original reviewer and timestamp; refreshes only the fields
        that describe the item's classification, since the edit changed those."""
        verdicts = self.verdicts(img_name)
        verdicts[item.key] = {
            **old_verdict,
            "kind": item.kind, "class_id": item.class_id,
            "conf": item.prediction.confidence if item.prediction else None,
            "iou": round(item.iou, 4) if item.iou is not None else None,
        }
        self.save_review_state()

    # ── Flags for a second look ────────────────────────────────────────────

    def flags(self, img_name):
        """The live flag history for an image: QueueItem.key -> list of flag entries, oldest first.

        A resolved entry is kept for the record; only the last entry of a key
        can be open.
        """
        return self._image_entry(img_name).setdefault("flags", {})

    def open_flag(self, img_name, key):
        """The open flag entry for key, or None when it has none or its last one is resolved."""
        history = self.flags(img_name).get(key)
        if history and not history[-1]["resolved"]:
            return history[-1]
        return None

    def open_flag_keys(self, img_name):
        """Keys with an open flag on an image, whether or not their item still exists."""
        return {key for key in self.flags(img_name) if self.open_flag(img_name, key)}

    def has_open_flags(self, img_name):
        """Whether an image holds any open flag, read without loading it."""
        entry = self.state._review_state.get("image", {}).get(img_name, {})
        return any(history and not history[-1]["resolved"]
                   for history in entry.get("flags", {}).values())

    def save_flag(self, img_name, item, comment, user):
        """Open a flag on item with comment, or replace the comment of its open flag, and save."""
        existing = self.open_flag(img_name, item.key)
        if existing is not None:
            existing["comment"] = comment
        else:
            self.flags(img_name).setdefault(item.key, []).append({
                "comment": comment, "kind": item.kind, "class_id": item.class_id,
                "by": user, "at": datetime.datetime.now().isoformat(timespec="seconds"),
                "resolved": False, "resolved_by": None, "resolved_at": None})
        self.save_review_state()

    def resolve_flag(self, img_name, key, user):
        """Mark the open flag on key resolved, keeping it in the history, and save."""
        entry = self.open_flag(img_name, key)
        if entry is None:
            return
        entry.update(resolved=True, resolved_by=user,
                     resolved_at=datetime.datetime.now().isoformat(timespec="seconds"))
        self.save_review_state()

    def carry_flags(self, img_name, old_key, new_key):
        """Move a key's flag history onto the key a geometry edit reclassified it to, and save."""
        flags = self.flags(img_name)
        if old_key in flags and new_key not in flags:
            flags[new_key] = flags.pop(old_key)
            self.save_review_state()

    def remove_verdict(self, img_name, key):
        """Remove a recorded verdict, if present, and save."""
        self.verdicts(img_name).pop(key, None)
        self.save_review_state()

    def update_img_status(self, img_name):
        """Recompute not_started/started/completed for img_name from every queue item, ignoring active filters."""
        s = self.state
        verdicts = self.verdicts(img_name)
        if s.document is None or not s.matches:
            total, reviewed = 0, 0
        else:
            all_items = build_queue(s.document, s.predictions, s.matches, verdicts)
            total = len(all_items)
            reviewed = sum(1 for item in all_items if item.key in verdicts)
        entry = self._image_entry(img_name)
        if reviewed == 0:
            entry["img_status"] = "not_started"
        elif total and reviewed >= total:
            entry["img_status"] = "completed"
        else:
            entry["img_status"] = "started"
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
        """Copy label files to .original/ before the first label write; idempotent."""
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

