"""Tests for yololabeler.label_io — YOLO label parsing and writing."""

import json

import pytest

from yololabeler.label_io import (
    write_detect_labels, write_segment_labels,
    format_detect_line, format_segment_line, parse_label_file, write_json_atomic,
)

IMG_W, IMG_H = 640, 480


# ── write_detect_labels ─────────────────────────────────────────────────────

class TestWriteDetectLabels:
    def test_roundtrip(self, tmp_path):
        p = str(tmp_path / "det.txt")
        boxes = [(192.0, 96.0, 448.0, 384.0, 0)]
        write_detect_labels(p, boxes, IMG_W, IMG_H)
        rows = parse_label_file(p, "box", IMG_W, IMG_H).rows
        assert len(rows) == 1 and rows[0].class_id == 0
        for a, b in zip(boxes[0][:4], (*rows[0].points[0], *rows[0].points[1])):
            assert pytest.approx(a, abs=0.1) == b

    def test_empty_removes_file(self, tmp_path):
        p = tmp_path / "det.txt"
        p.write_text("junk", encoding="utf-8")
        write_detect_labels(str(p), [], IMG_W, IMG_H)
        assert not p.exists()

    def test_empty_no_file_noop(self, tmp_path):
        p = str(tmp_path / "det.txt")
        write_detect_labels(p, [], IMG_W, IMG_H)  # should not raise


# ── write_segment_labels ────────────────────────────────────────────────────

class TestWriteSegmentLabels:
    def test_roundtrip(self, tmp_path):
        p = str(tmp_path / "seg.txt")
        polys = [([(0.0, 0.0), (640.0, 0.0), (320.0, 480.0)], 1)]
        write_segment_labels(p, polys, IMG_W, IMG_H)
        rows = parse_label_file(p, "polygon", IMG_W, IMG_H).rows
        assert len(rows) == 1 and rows[0].class_id == 1
        for (ax, ay), (bx, by) in zip(polys[0][0], rows[0].points):
            assert pytest.approx(ax, abs=0.1) == bx
            assert pytest.approx(ay, abs=0.1) == by

    def test_empty_removes_file(self, tmp_path):
        p = tmp_path / "seg.txt"
        p.write_text("junk", encoding="utf-8")
        write_segment_labels(str(p), [], IMG_W, IMG_H)
        assert not p.exists()


# ── format lines ────────────────────────────────────────────────────────────

class TestFormatLines:
    def test_detect_line_matches_writer_output(self):
        line = format_detect_line(10, 20, 30, 60, 2, 100, 200)
        assert line == "2 0.200000 0.200000 0.200000 0.200000"

    def test_segment_line(self):
        line = format_segment_line([(0, 0), (50, 0), (50, 100)], 1, 100, 200)
        assert line == "1 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000"


# ── parse_label_file ────────────────────────────────────────────────────────

class TestParseLabelFile:
    def test_missing_file_is_empty(self, tmp_path):
        result = parse_label_file(tmp_path / "none.txt", "box", 100, 100)
        assert result.rows == [] and result.rejected == []

    def test_reports_rejected_line_numbers(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\nbad line\n1 0.1 0.1 0.1\n", encoding="utf-8")
        result = parse_label_file(p, "box", 100, 100)
        assert [r.class_id for r in result.rows] == [0]
        assert result.rejected == [2, 3]
        assert result.rows[0].points == ((40.0, 40.0), (60.0, 60.0))
        assert result.rows[0].line == "0 0.5 0.5 0.2 0.2"

    def test_confidence_column(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = parse_label_file(p, "box", 100, 100, with_conf=True)
        assert result.rows[0].confidence == pytest.approx(0.9)

    def test_polygon_rows(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("3 0 0 0.5 0 0.5 0.5\n3 0 0 0.5\n", encoding="utf-8")
        result = parse_label_file(p, "polygon", 100, 200)
        assert result.rows[0].points == ((0.0, 0.0), (50.0, 0.0), (50.0, 100.0))
        assert result.rejected == [2]


# ── write_json_atomic ───────────────────────────────────────────────────────

class TestWriteJsonAtomic:
    def test_round_trip_and_no_temp_left(self, tmp_path):
        p = tmp_path / "s.json"
        write_json_atomic(p, {"a": 1})
        assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
        assert [f.name for f in tmp_path.iterdir()] == ["s.json"]
