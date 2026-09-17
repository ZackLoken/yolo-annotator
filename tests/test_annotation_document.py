"""Tests for yololabeler.annotation.document, the per-image annotation record."""

from pathlib import Path

import pytest

from yololabeler.annotation.document import (
    Document, load_document, new_annotation, save_document,
)

BOX = ((10.0, 20.0), (30.0, 60.0))
TRI = ((0.0, 0.0), (50.0, 0.0), (50.0, 100.0))


def paths(tmp_path):
    return (tmp_path / "detect.txt", tmp_path / "segment.txt", tmp_path / "side.json")


# ── new_annotation ──────────────────────────────────────────────────────────

class TestNewAnnotation:
    def test_assigns_id_and_created(self):
        a = new_annotation("box", BOX, 0, "zack")
        assert len(a.id) == 36 and a.created and a.source == "drawn"

    def test_accepted_carries_prediction(self):
        a = new_annotation("box", BOX, 0, "zack", source="accepted",
                           prediction_id="abc:3", confidence=0.9)
        assert a.prediction_id == "abc:3" and a.confidence == 0.9

    def test_ids_are_unique(self):
        assert new_annotation("box", BOX, 0, "").id != new_annotation("box", BOX, 0, "").id


# ── Document ────────────────────────────────────────────────────────────────

class TestDocument:
    def test_add_get_remove(self):
        doc = Document("a.jpg", 100, 200)
        a = new_annotation("box", BOX, 0, "z")
        doc.add(a)
        assert doc.get(a.id) is a
        assert doc.remove(a.id) is a
        with pytest.raises(KeyError):
            doc.get(a.id)

    def test_replace_keeps_id(self):
        doc = Document("a.jpg", 100, 200)
        a = new_annotation("polygon", TRI, 1, "z")
        doc.add(a)
        b = doc.replace(a.id, class_id=2)
        assert b.id == a.id and b.class_id == 2 and doc.get(a.id).class_id == 2

    def test_boxes_and_polygons_keep_order(self):
        doc = Document("a.jpg", 100, 200)
        p = new_annotation("polygon", TRI, 1, "z")
        b = new_annotation("box", BOX, 0, "z")
        doc.add(p)
        doc.add(b)
        assert doc.boxes() == [b] and doc.polygons() == [p]

    def test_snapshot_restore(self):
        doc = Document("a.jpg", 100, 200)
        a = new_annotation("box", BOX, 0, "z")
        doc.add(a)
        snap = doc.snapshot()
        doc.remove(a.id)
        doc.restore(snap)
        assert doc.annotations == [a]

    def test_label_lines(self):
        doc = Document("a.jpg", 100, 200)
        doc.add(new_annotation("box", BOX, 2, "z"))
        doc.add(new_annotation("polygon", TRI, 1, "z"))
        detect, segment = doc.label_lines()
        assert detect == ["2 0.200000 0.200000 0.200000 0.200000"]
        assert segment == ["1 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000"]


# ── save_document / load_document ───────────────────────────────────────────

class TestRoundTrip:
    def test_round_trip_keeps_provenance(self, tmp_path):
        d, s, side = paths(tmp_path)
        doc = Document("a.jpg", 100, 200)
        a = new_annotation("box", BOX, 2, "zack", source="accepted",
                           prediction_id="abc:0", confidence=0.8)
        doc.add(a)
        save_document(doc, d, s, side)
        loaded, rejected, _ = load_document("a.jpg", 100, 200, d, s, side)
        assert rejected == []
        b = loaded.annotations[0]
        assert (b.id, b.author, b.source, b.prediction_id, b.confidence) == \
            (a.id, "zack", "accepted", "abc:0", 0.8)
        # pytest.approx (9.0.2) rejects nested tuples; compare corner by corner.
        assert b.points[0] == pytest.approx(BOX[0])
        assert b.points[1] == pytest.approx(BOX[1])

    def test_corrupt_sidecar_is_quarantined_and_the_labels_still_load(self, tmp_path):
        d, s, side = paths(tmp_path)
        d.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        side.write_text("{not json", encoding="utf-8")
        loaded, rejected, moved = load_document("a.jpg", 100, 200, d, s, side)
        assert rejected == [] and len(loaded.annotations) == 1
        assert loaded.annotations[0].source == "unknown"
        assert moved and not side.exists() and Path(moved).exists()

    def test_empty_document_removes_files(self, tmp_path):
        d, s, side = paths(tmp_path)
        d.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        save_document(Document("a.jpg", 100, 200), d, s, side)
        assert not d.exists() and not side.exists()

    def test_line_without_record_is_unknown(self, tmp_path):
        d, s, side = paths(tmp_path)
        d.write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
        loaded, _, _ = load_document("a.jpg", 100, 200, d, s, side)
        a = loaded.annotations[0]
        assert a.source == "unknown" and a.author == "" and len(a.id) == 36

    def test_record_without_line_is_dropped(self, tmp_path):
        d, s, side = paths(tmp_path)
        doc = Document("a.jpg", 100, 200)
        doc.add(new_annotation("box", BOX, 2, "zack"))
        save_document(doc, d, s, side)
        d.unlink()
        loaded, _, _ = load_document("a.jpg", 100, 200, d, s, side)
        assert loaded.annotations == []

    def test_rejected_lines_reported(self, tmp_path):
        d, s, side = paths(tmp_path)
        d.write_text("0 0.5 0.5 0.2 0.2\nnope\n", encoding="utf-8")
        loaded, rejected, _ = load_document("a.jpg", 100, 200, d, s, side)
        assert len(loaded.annotations) == 1
        assert rejected == [f"{d}: line 2"]

    def test_identical_lines_keep_separate_records(self, tmp_path):
        d, s, side = paths(tmp_path)
        doc = Document("a.jpg", 100, 200)
        first = new_annotation("box", BOX, 2, "zack")
        second = new_annotation("box", BOX, 2, "nathan", source="accepted",
                                prediction_id="abc:0", confidence=0.7)
        doc.add(first)
        doc.add(second)
        save_document(doc, d, s, side)
        assert len(d.read_text(encoding="utf-8").splitlines()) == 2
        loaded, _, _ = load_document("a.jpg", 100, 200, d, s, side)
        assert [a.id for a in loaded.annotations] == [first.id, second.id]
        assert [a.author for a in loaded.annotations] == ["zack", "nathan"]
        assert loaded.remove(first.id).author == "zack"
        assert loaded.get(second.id).prediction_id == "abc:0"

    def test_legacy_authors_by_position(self, tmp_path):
        d, s, side = paths(tmp_path)
        d.write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        loaded, _, _ = load_document("a.jpg", 100, 200, d, s, side,
                                  legacy_authors=(["nathan", ""], []))
        assert [a.author for a in loaded.annotations] == ["nathan", ""]
        assert loaded.annotations[0].source == "unknown"
