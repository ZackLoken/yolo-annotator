"""Tests for ReviewEngine — headless review logic."""

import os
import tempfile

import pytest

from yololabeler.annotation.document import Document, new_annotation
from yololabeler.predictions.store import Prediction
from yololabeler.state import AppState
from yololabeler.review.engine import (
    DEFAULT_CONF_THRESHOLD, QueueItem, ReviewEngine, apply_accept, apply_reject,
    build_queue, match_document,
)


@pytest.fixture
def engine():
    state = AppState()
    state.images = ["img_001.jpg"]
    state._review_index = 0
    state._review_img_w = 640
    state._review_img_h = 480
    state.class_names = {0: "tree", 1: "shrub"}
    return ReviewEngine(state)


# ── Review state persistence ──────────────────────────────────────────────


class TestReviewStatePersistence:
    def test_save_and_load(self, engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.state.image_folder = tmpdir
            engine.state.state_dir = os.path.join(tmpdir, "state")
            os.makedirs(engine.state.state_dir, exist_ok=True)
            engine.state._review_state = {"image": {"a.jpg": {"img_status": "completed"}}}
            engine.save_review_state()
            engine.state._review_state = {}
            engine.load_review_state()
            assert engine.state._review_state["image"]["a.jpg"]["img_status"] == "completed"

    def test_load_missing_file(self, engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.state.image_folder = tmpdir
            engine.state.state_dir = os.path.join(tmpdir, "state")
            os.makedirs(engine.state.state_dir, exist_ok=True)
            engine.load_review_state()
            assert engine.state._review_state == {}

    def test_review_state_path_empty_folder(self, engine):
        engine.state.image_folder = ""
        assert engine.review_state_path() is None


# ── Image review status ───────────────────────────────────────────────────


class TestImageReviewStatus:
    def test_mark_and_check_reviewed(self, engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.state.image_folder = tmpdir
            engine.state.state_dir = os.path.join(tmpdir, "state")
            os.makedirs(engine.state.state_dir, exist_ok=True)
            engine.mark_image_reviewed("img_001.jpg")
            assert engine.is_image_reviewed("img_001.jpg")
            assert engine.get_image_review_status("img_001.jpg") == "completed"

    def test_not_reviewed(self, engine):
        assert not engine.is_image_reviewed("unknown.jpg")
        assert engine.get_image_review_status("unknown.jpg") == "not_started"


# ── Detection bounding boxes ──────────────────────────────────────────────


class TestMatchBbox:
    def test_gt_box(self, engine):
        engine.state._review_gt_boxes = [(100, 100, 200, 200, 0)]
        bbox = engine.match_bbox('box', 0, None, None)
        assert bbox == (100, 100, 200, 200)

    def test_pred_polygon(self, engine):
        engine.state._review_pred_polygons = [([(10, 10), (50, 10), (30, 50)], 0)]
        bbox = engine.match_bbox(None, None, 'polygon', 0)
        assert bbox == (10, 10, 50, 50)

    def test_combined_tp(self, engine):
        engine.state._review_gt_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_pred_boxes = [(90, 90, 210, 210, 0)]
        bbox = engine.match_bbox('box', 0, 'box', 0)
        assert bbox == (90, 90, 210, 210)

    def test_invalid_index(self, engine):
        engine.state._review_gt_boxes = []
        bbox = engine.match_bbox('box', 5, None, None)
        assert bbox == (0, 0, 640, 480)


class TestDetNormBbox:
    def test_box_pred(self, engine):
        engine.state._review_pred_boxes = [(160, 120, 480, 360, 0)]
        det = {'pred_type': 'box', 'pred_idx': 0, 'gt_type': None, 'gt_idx': None}
        result = engine.det_norm_bbox(det, 'pred')
        assert result is not None
        assert len(result) == 4
        assert abs(result[0] - 0.5) < 0.001  # cx
        assert abs(result[1] - 0.5) < 0.001  # cy

    def test_auto_fallback_to_gt(self, engine):
        engine.state._review_gt_boxes = [(0, 0, 640, 480, 0)]
        det = {'pred_type': None, 'pred_idx': None, 'gt_type': 'box', 'gt_idx': 0}
        result = engine.det_norm_bbox(det, 'auto')
        assert result is not None


# ── Reviewed entry lookup ─────────────────────────────────────────────────


class TestReviewedEntryLookup:
    def test_invalidate_and_rebuild(self, engine):
        engine.invalidate_reviewed_lookup()
        assert engine.state._reviewed_lookup == ("", {}, {})
        engine.build_reviewed_lookup("img_001.jpg")
        assert engine.state._reviewed_lookup[0] == "img_001.jpg"

    def test_find_reviewed_entry_none(self, engine):
        det = {'det_type': 'tp', 'pred_type': 'box', 'pred_idx': 0,
               'gt_type': 'box', 'gt_idx': 0}
        engine.state._review_pred_boxes = [(100, 100, 200, 200, 0)]
        result = engine.find_reviewed_entry(det, "img_001.jpg")
        assert result is None


# ── Detection list ────────────────────────────────────────────────────────


class TestRebuildReviewDetections:
    def test_empty_matches(self, engine):
        engine.rebuild_review_detections()
        assert engine.state._review_detections == []

    def test_with_tp(self, engine):
        engine.state._review_gt_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_pred_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_matches = {
            'tp': [('box', 0, 'box', 0, 0.95, 0, 0.99)],
            'fp': [],
            'fn': [],
        }
        engine.rebuild_review_detections()
        assert len(engine.state._review_detections) == 1
        assert engine.state._review_detections[0]['det_type'] == 'tp'

    def test_filter_by_class(self, engine):
        engine.state._review_gt_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_pred_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_matches = {
            'tp': [('box', 0, 'box', 0, 0.95, 0, 0.99)],
            'fp': [],
            'fn': [],
        }
        engine.state._review_filter_class = 1  # shrub, not tree
        engine.rebuild_review_detections()
        assert len(engine.state._review_detections) == 0

    def test_filter_by_type(self, engine):
        engine.state._review_gt_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_pred_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_matches = {
            'tp': [('box', 0, 'box', 0, 0.95, 0, 0.99)],
            'fp': [],
            'fn': [],
        }
        engine.state._review_filter_type = "fp"
        engine.rebuild_review_detections()
        assert len(engine.state._review_detections) == 0


# ── Record & check ────────────────────────────────────────────────────────


class TestRecordDetectionAction:
    def test_records_action(self, engine):
        engine.state._review_gt_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_pred_boxes = [(100, 100, 200, 200, 0)]
        det = {
            'det_type': 'tp', 'class_id': 0, 'conf': 0.95, 'iou': 0.9,
            'gt_type': 'box', 'gt_idx': 0,
            'pred_type': 'box', 'pred_idx': 0,
            'bbox': (100, 100, 200, 200),
        }
        engine.record_detection_action(det, 'accepted')
        per_image = engine.state._review_state.get("image", {})
        img_data = per_image.get("img_001.jpg")
        assert img_data is not None
        assert len(img_data["detections"]) == 1
        assert img_data["detections"][0]["action"] == "accepted"


class TestCheckImageReviewComplete:
    def test_marks_complete(self, engine):
        engine.state._review_gt_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_pred_boxes = [(100, 100, 200, 200, 0)]
        engine.state._review_matches = {
            'tp': [('box', 0, 'box', 0, 0.95, 0, 0.99)],
            'fp': [],
            'fn': [],
        }
        # Record action for the single detection
        det = {
            'det_type': 'tp', 'class_id': 0, 'conf': 0.95, 'iou': 0.9,
            'gt_type': 'box', 'gt_idx': 0,
            'pred_type': 'box', 'pred_idx': 0,
        }
        engine.record_detection_action(det, 'accepted')
        engine.check_image_review_complete()
        per_image = engine.state._review_state.get("image", {})
        assert per_image["img_001.jpg"]["img_status"] == "completed"

    def test_not_complete_when_detections_remain(self, engine):
        engine.state._review_matches = {
            'tp': [('box', 0, 'box', 0, 0.95, 0, 0.99)],
            'fp': [('box', 1, 1, 0.8)],
            'fn': [],
        }
        engine.check_image_review_complete()
        per_image = engine.state._review_state.get("image", {})
        assert per_image.get("img_001.jpg") is None


# ── Label backup ──────────────────────────────────────────────────────────


class TestBackupOriginalLabels:
    def test_creates_backup(self, engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = engine.state
            s.image_folder = tmpdir
            s.state_dir = os.path.join(tmpdir, "state")
            s.detect_dir = os.path.join(tmpdir, "labels", "detect")
            s.segment_dir = os.path.join(tmpdir, "labels", "segment")
            os.makedirs(s.detect_dir, exist_ok=True)
            os.makedirs(s.segment_dir, exist_ok=True)
            os.makedirs(s.state_dir, exist_ok=True)
            # Create a dummy label file
            with open(os.path.join(s.detect_dir, "test.txt"), "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
            engine.backup_original_labels()
            backup_path = os.path.join(s.detect_dir, ".original", "test.txt")
            assert os.path.exists(backup_path)
            assert s._review_state.get("labels_backed_up")

    def test_skips_if_already_backed_up(self, engine):
        engine.state._review_state["labels_backed_up"] = True
        engine.backup_original_labels()  # should be a no-op


# ── Save GT ───────────────────────────────────────────────────────────────


class TestSaveGt:
    def test_writes_gt_files(self, engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = engine.state
            s.detect_dir = os.path.join(tmpdir, "detect")
            s.segment_dir = os.path.join(tmpdir, "segment")
            os.makedirs(s.detect_dir, exist_ok=True)
            os.makedirs(s.segment_dir, exist_ok=True)
            s._review_gt_boxes = [(100, 100, 200, 200, 0)]
            s._review_gt_polygons = [([(10, 10), (50, 10), (30, 50)], 0)]
            engine.save_gt()
            assert os.path.exists(os.path.join(s.detect_dir, "img_001.txt"))
            assert os.path.exists(os.path.join(s.segment_dir, "img_001.txt"))


# ── Document-based queue and verdicts ───────────────────────────────────────


def pred(pid, x1, y1, x2, y2, cid=0, conf=0.9):
    """Build a box Prediction for tests, deriving line_index from the id."""
    return Prediction(pid, "box", ((x1, y1), (x2, y2)), cid, conf, int(pid.split(":")[1]))


@pytest.fixture
def scene():
    """One matched box, one model miss, one false positive, one low-confidence pred."""
    doc = Document("img_001.jpg", 640, 480)
    doc.add(new_annotation("box", ((10, 10), (110, 110)), 0, "z"))
    doc.add(new_annotation("box", ((300, 300), (400, 400)), 0, "z"))
    preds = [pred("h:0", 12, 12, 112, 112),
             pred("h:1", 500, 20, 560, 80),
             pred("h:2", 500, 300, 560, 360, conf=0.3)]
    return doc, preds


# ── match_document / build_queue ────────────────────────────────────────────

class TestQueue:
    def test_queue_order_and_keys(self, scene):
        doc, preds = scene
        matches = match_document(doc, preds, 0.6, 0.5)
        queue = build_queue(doc, preds, matches, {})
        assert [q.kind for q in queue] == ["fp", "fn", "tp"]
        assert queue[0].key == "h:1"
        assert queue[1].key == doc.annotations[1].id
        assert queue[2].key == "h:0" and queue[2].annotation is doc.annotations[0]
        assert queue[2].iou == pytest.approx(0.9238, abs=0.001)

    def test_low_confidence_is_absent(self, scene):
        doc, preds = scene
        queue = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})
        assert all(q.key != "h:2" for q in queue)

    def test_threshold_change_reveals_it(self, scene):
        doc, preds = scene
        queue = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.25), {})
        assert any(q.key == "h:2" and q.kind == "fp" for q in queue)

    def test_filters(self, scene):
        doc, preds = scene
        matches = match_document(doc, preds, 0.6, 0.5)
        verdicts = {"h:1": {"action": "rejected"}}
        assert [q.kind for q in build_queue(doc, preds, matches, verdicts, filter_type="fn")] == ["fn"]
        assert [q.key for q in build_queue(doc, preds, matches, verdicts, filter_status="reviewed")] == ["h:1"]
        assert len(build_queue(doc, preds, matches, verdicts, filter_status="not_reviewed")) == 2
        assert build_queue(doc, preds, matches, verdicts, filter_class=7) == []

    def test_polygon_prediction_matches_polygon_annotation(self):
        doc = Document("a.jpg", 100, 100)
        doc.add(new_annotation("polygon", ((0, 0), (50, 0), (50, 50), (0, 50)), 1, "z"))
        p = Prediction("h:0", "polygon", ((1, 1), (50, 0), (50, 50), (0, 50)), 1, 0.8, 0)
        queue = build_queue(doc, [p], match_document(doc, [p], 0.6, 0.5), {})
        assert [q.kind for q in queue] == ["tp"]


# ── apply_accept / apply_reject ─────────────────────────────────────────────

class TestActions:
    def test_accept_fp_inserts_annotation_with_provenance(self, scene):
        doc, preds = scene
        item = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})[0]
        action, created = apply_accept(doc, item, "ren")
        assert action == "accepted"
        assert created in doc.annotations
        assert created.source == "accepted" and created.prediction_id == "h:1"
        assert created.confidence == pytest.approx(0.9) and created.author == "ren"
        assert created.points == ((500.0, 20.0), (560.0, 80.0)) and created.class_id == 0

    def test_accept_tp_and_fn_change_nothing(self, scene):
        doc, preds = scene
        queue = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})
        before = list(doc.annotations)
        assert apply_accept(doc, queue[1], "ren") == ("kept", None)
        assert apply_accept(doc, queue[2], "ren") == ("confirmed", None)
        assert doc.annotations == before

    def test_reject_fp_changes_nothing(self, scene):
        doc, preds = scene
        item = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})[0]
        assert apply_reject(doc, item) == ("rejected", None)
        assert len(doc.annotations) == 2

    def test_reject_tp_and_fn_delete(self, scene):
        doc, preds = scene
        queue = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})
        action, removed = apply_reject(doc, queue[2])
        assert action == "rejected" and removed not in doc.annotations
        action, removed = apply_reject(doc, queue[1])
        assert removed not in doc.annotations and doc.annotations == []


# ── verdict persistence ─────────────────────────────────────────────────────

class TestVerdicts:
    def test_record_and_remove(self, engine, tmp_path, scene):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        doc, preds = scene
        item = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})[0]
        engine.record_verdict("img_001.jpg", item, "rejected", "ren")
        v = engine.verdicts("img_001.jpg")["h:1"]
        assert v["action"] == "rejected" and v["by"] == "ren" and v["class_id"] == 0
        assert v["conf"] == pytest.approx(0.9) and v["iou"] is None and v["at"]
        engine.save_review_state()
        engine.load_review_state()
        assert "h:1" in engine.verdicts("img_001.jpg")
        engine.remove_verdict("img_001.jpg", "h:1")
        assert engine.verdicts("img_001.jpg") == {}

    def test_conf_threshold_persists(self, engine, tmp_path):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        assert engine.conf_threshold == DEFAULT_CONF_THRESHOLD
        engine.conf_threshold = 0.3
        engine.load_review_state()
        assert engine.conf_threshold == pytest.approx(0.3)

    def test_corrupt_state_is_quarantined(self, engine, tmp_path):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        path = os.path.join(engine.state.state_dir, "review_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{")
        moved = engine.load_review_state()
        assert moved and moved.startswith(path + ".corrupt-")
        assert engine.state._review_state == {}


# ── migrate_centre_entries ──────────────────────────────────────────────────

class TestMigrateCentreEntries:
    def test_moves_matching_entries_and_drops_the_rest(self, engine, tmp_path):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        engine.state._review_state = {"image": {"img_001.jpg": {
            "img_status": "started",
            "detections": [
                {"match_type": "FP", "action": "rejected", "reviewed_by": "zack",
                 "class_id": 0, "pred_bbox_norm": [0.828125, 0.104167, 0.09375, 0.125],
                 "gt_bbox_norm": None, "iou": None, "conf": 0.9},
                {"match_type": "FP", "action": "accepted", "reviewed_by": "zack",
                 "class_id": 0, "pred_bbox_norm": [0.1, 0.1, 0.05, 0.05],
                 "gt_bbox_norm": None, "iou": None, "conf": 0.7},
            ]}}}
        preds = [pred("h:1", 500, 20, 560, 80)]
        dropped = engine.migrate_centre_entries("img_001.jpg", preds, 640, 480)
        assert dropped == 1
        v = engine.verdicts("img_001.jpg")
        assert v["h:1"]["action"] == "rejected" and v["h:1"]["by"] == "zack"
        assert "detections" not in engine.state._review_state["image"]["img_001.jpg"]

    def test_noop_without_legacy_entries(self, engine):
        assert engine.migrate_centre_entries("img_001.jpg", [], 640, 480) == 0
