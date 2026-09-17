"""Tests for ReviewEngine — headless review logic."""

import os
import tempfile

import pytest

from yololabeler.annotation.document import Document, new_annotation
from yololabeler.predictions.store import Prediction, write_manifest
from yololabeler.state import AppState
from yololabeler.review.engine import (
    DEFAULT_CONF_THRESHOLD, QueueItem, ReviewEngine, apply_accept, apply_reject,
    build_queue, flag_markers, match_document, shape_statuses, spatial_order,
    unmatched_prediction_ids,
)


@pytest.fixture
def engine():
    state = AppState()
    state.images = ["img_001.jpg"]
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


def of_kind(queue, kind):
    """The one queue item of that kind."""
    (item,) = [q for q in queue if q.kind == kind]
    return item


# ── match_document / build_queue ────────────────────────────────────────────

class TestQueue:
    def test_queue_keys(self, scene):
        doc, preds = scene
        matches = match_document(doc, preds, 0.6, 0.5)
        queue = build_queue(doc, preds, matches, {})
        assert of_kind(queue, "fp").key == "h:1"
        assert of_kind(queue, "fn").key == doc.annotations[1].id
        tp = of_kind(queue, "tp")
        assert tp.key == "h:0" and tp.annotation is doc.annotations[0]
        assert tp.iou == pytest.approx(0.9238, abs=0.001)

    def test_queue_follows_the_nearest_neighbour_path_from_the_top_left(self, scene):
        doc, preds = scene
        queue = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})
        # tp centre (62, 62) is nearest the corner; fn (350, 350) is nearer to it than fp (530, 50).
        assert [q.kind for q in queue] == ["tp", "fn", "fp"]

    def test_status_filter_does_not_reorder_the_path(self, scene):
        doc, preds = scene
        matches = match_document(doc, preds, 0.6, 0.5)
        verdicts = {doc.annotations[1].id: {"action": "accepted"}}
        queue = build_queue(doc, preds, matches, verdicts, filter_status="not_reviewed")
        assert [q.kind for q in queue] == ["tp", "fp"]

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


# ── spatial_order ───────────────────────────────────────────────────────────

class TestSpatialOrder:
    def test_a_cluster_is_visited_before_moving_on(self):
        far = [pred("h:0", 900, 900, 910, 910), pred("h:1", 0, 0, 10, 10),
               pred("h:2", 600, 20, 610, 30), pred("h:3", 30, 0, 40, 10),
               pred("h:4", 15, 25, 25, 35)]
        items = [QueueItem("fp", p, None, None) for p in far]
        assert [q.key for q in spatial_order(items)] == ["h:1", "h:4", "h:3", "h:2", "h:0"]

    def test_empty(self):
        assert spatial_order([]) == []


# ── shape_statuses ──────────────────────────────────────────────────────────

class TestUnmatchedPredictionIds:
    def test_fps_only_and_empty_before_matching(self, scene):
        doc, preds = scene
        assert unmatched_prediction_ids(preds, {}) == set()
        matches = match_document(doc, preds, 0.6, 0.5)
        assert unmatched_prediction_ids(preds, matches) == {"h:1"}


class TestShapeStatuses:
    def test_none_without_matches(self, scene):
        doc, _ = scene
        assert shape_statuses(doc, [], {}, {}) is None

    def test_a_match_shares_its_status_with_its_annotation(self, scene):
        doc, preds = scene
        matches = match_document(doc, preds, 0.6, 0.5)
        verdicts = {"h:0": {"action": "accepted"}, "h:1": {"action": "rejected"}}
        assert shape_statuses(doc, preds, matches, verdicts) == {
            "h:0": "accepted", doc.annotations[0].id: "accepted",
            "h:1": "rejected", doc.annotations[1].id: "not_reviewed"}

    def test_low_confidence_prediction_has_no_status(self, scene):
        doc, preds = scene
        statuses = shape_statuses(doc, preds, match_document(doc, preds, 0.6, 0.5), {})
        assert "h:2" not in statuses


# ── apply_accept / apply_reject ─────────────────────────────────────────────

class TestActions:
    def test_accept_fp_inserts_annotation_with_provenance(self, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fp")
        action, created = apply_accept(doc, item, "ren")
        assert action == "accepted"
        assert created in doc.annotations
        assert created.source == "accepted" and created.prediction_id == "h:1"
        assert created.confidence == pytest.approx(0.9) and created.author == "ren"
        assert created.points == ((500.0, 20.0), (560.0, 80.0)) and created.class_id == 0

    def test_accept_clamps_an_edge_of_frame_prediction_to_the_image(self, scene):
        doc, preds = scene
        preds = preds[:1] + [pred("h:1", -20, -10, 700, 500)]
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fp")
        _, created = apply_accept(doc, item, "ren")
        assert created.points == ((0.0, 0.0), (640.0, 480.0))

    def test_accept_tp_and_fn_change_nothing(self, scene):
        doc, preds = scene
        queue = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})
        before = list(doc.annotations)
        assert apply_accept(doc, of_kind(queue, "fn"), "ren") == ("accepted", None)
        assert apply_accept(doc, of_kind(queue, "tp"), "ren") == ("accepted", None)
        assert doc.annotations == before

    def test_reject_fp_changes_nothing(self, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fp")
        assert apply_reject(doc, item) == ("rejected", None)
        assert len(doc.annotations) == 2

    def test_reject_tp_and_fn_delete(self, scene):
        doc, preds = scene
        queue = build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {})
        action, removed = apply_reject(doc, of_kind(queue, "tp"))
        assert action == "rejected" and removed not in doc.annotations
        action, removed = apply_reject(doc, of_kind(queue, "fn"))
        assert removed not in doc.annotations and doc.annotations == []


# ── verdict persistence ─────────────────────────────────────────────────────

class TestVerdicts:
    def test_record_and_remove(self, engine, tmp_path, scene):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fp")
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
        assert engine.conf_threshold == DEFAULT_CONF_THRESHOLD == pytest.approx(0.25)
        engine.conf_threshold = 0.3
        engine.load_review_state()
        assert engine.conf_threshold == pytest.approx(0.3)

    def test_conf_threshold_defaults_to_imported_min(self, engine, tmp_path):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        preds_dir = tmp_path / "predictions"
        os.makedirs(preds_dir)
        write_manifest(str(preds_dir), {"min_conf": 0.4})
        assert engine.conf_threshold == pytest.approx(0.4)

    def test_explicit_conf_threshold_overrides_imported_min(self, engine, tmp_path):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        preds_dir = tmp_path / "predictions"
        os.makedirs(preds_dir)
        write_manifest(str(preds_dir), {"min_conf": 0.25})
        engine.conf_threshold = 0.6
        assert engine.conf_threshold == pytest.approx(0.6)

    def test_reimport_resets_a_typed_threshold_to_the_new_min(self, engine, tmp_path):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        preds_dir = tmp_path / "predictions"
        os.makedirs(preds_dir)
        write_manifest(str(preds_dir), {"min_conf": 0.4, "imported_at": "2026-09-15T08:00:00"})
        engine.conf_threshold = 0.6
        write_manifest(str(preds_dir), {"min_conf": 0.35, "imported_at": "2026-09-16T08:00:00"})
        assert engine.conf_threshold == pytest.approx(0.35)

    def test_unstamped_saved_threshold_yields_to_imported_min(self, engine, tmp_path):
        engine.state.image_folder = str(tmp_path)
        engine.state.state_dir = str(tmp_path / "state")
        os.makedirs(engine.state.state_dir)
        preds_dir = tmp_path / "predictions"
        os.makedirs(preds_dir)
        write_manifest(str(preds_dir), {"min_conf": 0.35, "imported_at": "2026-09-16T08:00:00"})
        engine.state._review_state = {"settings": {"conf_threshold": 0.5}}
        assert engine.conf_threshold == pytest.approx(0.35)

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


# ── update_img_status ───────────────────────────────────────────────────────

class TestImgStatus:
    def test_freshly_created_entry_is_not_started(self, engine):
        engine.update_img_status("img_001.jpg")
        assert engine.state._review_state["image"]["img_001.jpg"]["img_status"] == "not_started"

    def test_one_verdict_of_several_gives_started(self, engine, scene):
        doc, preds = scene
        engine.state.document = doc
        engine.state.predictions = preds
        engine.state.matches = match_document(doc, preds, 0.6, 0.5)
        queue = build_queue(doc, preds, engine.state.matches, {})
        engine.record_verdict("img_001.jpg", queue[0], "rejected", "ren")
        engine.update_img_status("img_001.jpg")
        assert engine.state._review_state["image"]["img_001.jpg"]["img_status"] == "started"

    def test_every_item_reviewed_gives_completed(self, engine, scene):
        doc, preds = scene
        engine.state.document = doc
        engine.state.predictions = preds
        engine.state.matches = match_document(doc, preds, 0.6, 0.5)
        queue = build_queue(doc, preds, engine.state.matches, {})
        for item in queue:
            engine.record_verdict("img_001.jpg", item, "rejected", "ren")
        engine.update_img_status("img_001.jpg")
        assert engine.state._review_state["image"]["img_001.jpg"]["img_status"] == "completed"

    def test_filtering_has_no_bearing_on_the_computed_status(self, engine, scene):
        doc, preds = scene
        engine.state.document = doc
        engine.state.predictions = preds
        engine.state.matches = match_document(doc, preds, 0.6, 0.5)
        full_queue = build_queue(doc, preds, engine.state.matches, {})
        engine.record_verdict("img_001.jpg", full_queue[0], "rejected", "ren")
        verdicts = engine.verdicts("img_001.jpg")
        # A "reviewed" filter would make this look 1-of-1 complete; update_img_status must not use it.
        filtered = build_queue(doc, preds, engine.state.matches, verdicts,
                               filter_status="reviewed")
        assert len(filtered) == 1 and len(full_queue) == 3
        engine.update_img_status("img_001.jpg")
        assert engine.state._review_state["image"]["img_001.jpg"]["img_status"] == "started"


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


# ── flags ───────────────────────────────────────────────────────────────────

@pytest.fixture
def flag_engine(engine, tmp_path):
    engine.state.image_folder = str(tmp_path)
    engine.state.state_dir = str(tmp_path / "state")
    os.makedirs(engine.state.state_dir)
    return engine


def flag_item(engine, item, comment, user):
    engine.save_flag("img_001.jpg", item.key, item.kind, item.class_id, comment, user)


class TestFlags:
    def test_save_opens_a_flag_without_a_verdict(self, flag_engine, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fp")
        flag_item(flag_engine, item, "maybe a leaf", "ren")
        flag = flag_engine.open_flag("img_001.jpg", item.key)
        assert flag["comment"] == "maybe a leaf" and flag["by"] == "ren" and flag["at"]
        assert flag["kind"] == "fp" and not flag["resolved"] and flag["edited_by"] is None
        assert flag_engine.verdicts("img_001.jpg") == {}
        assert flag_engine.open_flag_keys("img_001.jpg") == {item.key}

    def test_editing_keeps_the_flagger_and_records_the_editor(self, flag_engine, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fn")
        flag_item(flag_engine, item, "", "ren")
        flag_item(flag_engine, item, "two burs?", "scott")
        (entry,) = flag_engine.flags("img_001.jpg")[item.key]
        assert entry["comment"] == "two burs?" and entry["by"] == "ren"
        assert entry["edited_by"] == "scott" and entry["edited_at"]

    def test_saving_an_unchanged_comment_records_no_edit(self, flag_engine, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fn")
        flag_item(flag_engine, item, "same", "ren")
        flag_item(flag_engine, item, "same", "scott")
        assert flag_engine.open_flag("img_001.jpg", item.key)["edited_by"] is None

    def test_resolve_keeps_the_record_and_a_new_flag_starts_a_new_entry(self, flag_engine, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "tp")
        flag_item(flag_engine, item, "check", "ren")
        flag_engine.resolve_flag("img_001.jpg", item.key, "zack")
        assert flag_engine.open_flag("img_001.jpg", item.key) is None
        (entry,) = flag_engine.flags("img_001.jpg")[item.key]
        assert entry["resolved"] and entry["resolved_by"] == "zack" and entry["resolved_at"]
        assert entry["resolved_note"] is None
        flag_item(flag_engine, item, "again", "ren")
        assert len(flag_engine.flags("img_001.jpg")[item.key]) == 2
        assert flag_engine.open_flag("img_001.jpg", item.key)["comment"] == "again"

    def test_resolve_records_a_note(self, flag_engine, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fn")
        flag_item(flag_engine, item, "", "ren")
        flag_engine.resolve_flag("img_001.jpg", item.key, "ren", note="rejected")
        assert flag_engine.flags("img_001.jpg")[item.key][0]["resolved_note"] == "rejected"

    def test_flags_persist_and_has_open_flags_reads_them(self, flag_engine, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fp")
        flag_item(flag_engine, item, "x", "ren")
        flag_engine.load_review_state()
        assert flag_engine.has_open_flags("img_001.jpg")
        assert not flag_engine.has_open_flags("other.jpg")
        flag_engine.resolve_flag("img_001.jpg", item.key, "zack")
        assert not flag_engine.has_open_flags("img_001.jpg")

    def test_carry_flags_moves_the_history_to_the_new_key(self, flag_engine, scene):
        doc, preds = scene
        item = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "fn")
        flag_item(flag_engine, item, "x", "ren")
        flag_engine.carry_flags("img_001.jpg", item.key, "h:9")
        assert flag_engine.open_flag_keys("img_001.jpg") == {"h:9"}

    def test_flag_key_prefers_an_annotation_flagged_before_predictions(self, flag_engine, scene):
        doc, preds = scene
        tp = of_kind(build_queue(doc, preds, match_document(doc, preds, 0.6, 0.5), {}), "tp")
        assert flag_engine.flag_key("img_001.jpg", tp) == tp.key
        flag_engine.save_flag("img_001.jpg", tp.annotation.id, None, 0, "", "ren")
        assert flag_engine.flag_key("img_001.jpg", tp) == tp.annotation.id

    def test_flagged_status_filter_keeps_only_open_flagged_items(self, scene):
        doc, preds = scene
        matches = match_document(doc, preds, 0.6, 0.5)
        queue = build_queue(doc, preds, matches, {}, filter_status="flagged", flagged={"h:1"})
        assert [q.key for q in queue] == ["h:1"]
        by_annotation = build_queue(doc, preds, matches, {}, filter_status="flagged",
                                    flagged={doc.annotations[0].id})
        assert [q.key for q in by_annotation] == ["h:0"]

    def test_markers_sit_on_the_annotation_when_there_is_one(self, scene):
        doc, preds = scene
        matches = match_document(doc, preds, 0.6, 0.5)
        markers = flag_markers(doc, preds, matches, {"h:0", "h:1", "gone"})
        assert markers == {"h:0": doc.annotations[0].points, "h:1": preds[1].points}

    def test_markers_without_matches_show_annotation_flags_only(self, scene):
        doc, preds = scene
        ann = doc.annotations[1]
        assert flag_markers(doc, preds, {}, {ann.id, "h:1"}) == {ann.id: ann.points}
