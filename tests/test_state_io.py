"""Tests for yololabeler.state_io, annotation_stats.json access and quarantine."""

import json

from yololabeler.state_io import AnnotationStats, read_json_or_quarantine


# ── read_json_or_quarantine ─────────────────────────────────────────────────

class TestReadJsonOrQuarantine:
    def test_missing(self, tmp_path):
        assert read_json_or_quarantine(tmp_path / "x.json") == (None, None)

    def test_valid(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        assert read_json_or_quarantine(p) == ({"a": 1}, None)

    def test_corrupt_is_renamed_not_deleted(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text('{"a": ', encoding="utf-8")
        data, moved = read_json_or_quarantine(p)
        assert data is None
        assert not p.exists()
        assert moved.startswith(str(tmp_path / "x.json.corrupt-"))
        assert open(moved, encoding="utf-8").read() == '{"a": '


# ── AnnotationStats ─────────────────────────────────────────────────────────

class TestAnnotationStats:
    def test_defaults(self, tmp_path):
        stats, moved = AnnotationStats.load(tmp_path / "s.json")
        assert moved is None
        assert stats.image_status("a.jpg") == "unannotated"
        assert not stats.is_blind("a.jpg")
        assert stats.completion("a.jpg") is None
        assert stats.sessions == []

    def test_status_blind_completion_round_trip(self, tmp_path):
        p = tmp_path / "s.json"
        stats, _ = AnnotationStats.load(p)
        stats.set_image_status("a.jpg", "complete")
        stats.set_blind("a.jpg", True)
        stats.set_completion("a.jpg", by="ren", blind=True, annotation_count=4, model=None)
        stats.save(p)
        again, _ = AnnotationStats.load(p)
        assert again.image_status("a.jpg") == "complete"
        assert again.is_blind("a.jpg")
        rec = again.completion("a.jpg")
        assert rec["by"] == "ren" and rec["blind"] and rec["annotation_count"] == 4
        assert rec["model"] is None and rec["at"]

    def test_set_blind_off(self, tmp_path):
        stats, _ = AnnotationStats.load(tmp_path / "s.json")
        stats.set_blind("a.jpg", True)
        stats.set_blind("a.jpg", False)
        assert not stats.is_blind("a.jpg")

    def test_clear_completion(self, tmp_path):
        stats, _ = AnnotationStats.load(tmp_path / "s.json")
        stats.set_completion("a.jpg", "ren", False, 1, "nathan_v15")
        stats.clear_completion("a.jpg")
        assert stats.completion("a.jpg") is None

    def test_pop_legacy_authors(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"sessions": [], "image_status": {},
                                 "annotation_authors": {"a.jpg": {"boxes": ["n"], "polygons": []}}}),
                     encoding="utf-8")
        stats, _ = AnnotationStats.load(p)
        assert stats.pop_legacy_authors("a.jpg") == (["n"], [])
        assert stats.pop_legacy_authors("a.jpg") is None
        stats.save(p)
        assert "annotation_authors" not in json.loads(p.read_text(encoding="utf-8"))

    def test_old_images_block_migrates_complete(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"images": {"a.jpg": {"status": "complete"}}}), encoding="utf-8")
        stats, _ = AnnotationStats.load(p)
        assert stats.image_status("a.jpg") == "complete"
        assert "images" not in stats.data
