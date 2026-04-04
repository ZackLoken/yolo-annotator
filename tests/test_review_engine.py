"""Tests for ReviewEngine — headless review logic."""

import json
import os
import tempfile

import pytest

from yololabeler.state import AppState
from yololabeler.review.engine import ReviewEngine


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
