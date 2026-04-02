"""Tests for yololabeler.label_io — YOLO label parsing and writing."""

import pytest

from yololabeler.label_io import (
    parse_detect_labels, parse_segment_labels,
    parse_detect_predictions, parse_segment_predictions,
    write_detect_labels, write_segment_labels,
)

IMG_W, IMG_H = 640, 480


# ── parse_detect_labels ─────────────────────────────────────────────────────

class TestParseDetectLabels:
    def test_basic(self, tmp_path):
        p = tmp_path / "det.txt"
        # class 0, centred at (0.5, 0.5), width 0.4, height 0.6
        p.write_text("0 0.500000 0.500000 0.400000 0.600000\n", encoding="utf-8")
        boxes, cids = parse_detect_labels(str(p), IMG_W, IMG_H)
        assert len(boxes) == 1
        x1, y1, x2, y2, cls = boxes[0]
        assert cls == 0
        assert pytest.approx(x1) == 192.0  # 320 - 128
        assert pytest.approx(y1) == 96.0   # 240 - 144
        assert pytest.approx(x2) == 448.0  # 320 + 128
        assert pytest.approx(y2) == 384.0  # 240 + 144
        assert cids == {0}

    def test_multiple_classes(self, tmp_path):
        p = tmp_path / "det.txt"
        p.write_text(
            "0 0.5 0.5 0.2 0.2\n"
            "3 0.1 0.1 0.1 0.1\n",
            encoding="utf-8",
        )
        boxes, cids = parse_detect_labels(str(p), IMG_W, IMG_H)
        assert len(boxes) == 2
        assert cids == {0, 3}

    def test_missing_file(self, tmp_path):
        boxes, cids = parse_detect_labels(str(tmp_path / "nope.txt"), IMG_W, IMG_H)
        assert boxes == []
        assert cids == set()

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "det.txt"
        p.write_text(
            "0 0.5 0.5 0.2\n"        # too few fields
            "0 0.5 0.5 0.2 0.2 0.1\n" # too many fields
            "0 0.5 0.5 0.2 0.2\n"     # correct
            "\n",                       # blank
            encoding="utf-8",
        )
        boxes, _ = parse_detect_labels(str(p), IMG_W, IMG_H)
        assert len(boxes) == 1

    def test_empty_file(self, tmp_path):
        p = tmp_path / "det.txt"
        p.write_text("", encoding="utf-8")
        boxes, cids = parse_detect_labels(str(p), IMG_W, IMG_H)
        assert boxes == []


# ── parse_segment_labels ────────────────────────────────────────────────────

class TestParseSegmentLabels:
    def test_triangle(self, tmp_path):
        p = tmp_path / "seg.txt"
        # class 1, triangle: (0,0) (1,0) (0.5,1)
        p.write_text("1 0.0 0.0 1.0 0.0 0.5 1.0\n", encoding="utf-8")
        polys, cids = parse_segment_labels(str(p), IMG_W, IMG_H)
        assert len(polys) == 1
        pts, cls = polys[0]
        assert cls == 1
        assert len(pts) == 3
        assert pts[0] == (0.0, 0.0)
        assert pts[1] == (640.0, 0.0)
        assert pts[2] == (320.0, 480.0)
        assert cids == {1}

    def test_missing_file(self, tmp_path):
        polys, cids = parse_segment_labels(str(tmp_path / "x.txt"), IMG_W, IMG_H)
        assert polys == []
        assert cids == set()

    def test_too_few_coords_skipped(self, tmp_path):
        p = tmp_path / "seg.txt"
        p.write_text(
            "0 0.1 0.2 0.3 0.4\n"  # only 2 vertices (4 vals) — need >= 3 (6 vals)
            "0 0.1 0.2 0.3 0.4 0.5 0.6\n",  # valid triangle
            encoding="utf-8",
        )
        polys, _ = parse_segment_labels(str(p), IMG_W, IMG_H)
        assert len(polys) == 1

    def test_even_total_parts_skipped(self, tmp_path):
        """Total parts must be odd (1 class_id + even number of coords)."""
        p = tmp_path / "seg.txt"
        # 8 parts total (even) → class_id + 7 coords → skipped
        p.write_text("0 0.1 0.2 0.3 0.4 0.5 0.6 0.7\n", encoding="utf-8")
        polys, _ = parse_segment_labels(str(p), IMG_W, IMG_H)
        assert len(polys) == 0


# ── parse_detect_predictions ────────────────────────────────────────────────

class TestParseDetectPredictions:
    def test_basic(self, tmp_path):
        p = tmp_path / "pred.txt"
        p.write_text("2 0.95 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        preds, cids = parse_detect_predictions(str(p), IMG_W, IMG_H)
        assert len(preds) == 1
        x1, y1, x2, y2, cls, conf = preds[0]
        assert cls == 2
        assert pytest.approx(conf) == 0.95
        assert cids == {2}

    def test_missing_file(self, tmp_path):
        preds, cids = parse_detect_predictions(str(tmp_path / "x.txt"), IMG_W, IMG_H)
        assert preds == []


# ── parse_segment_predictions ───────────────────────────────────────────────

class TestParseSegmentPredictions:
    def test_basic(self, tmp_path):
        p = tmp_path / "pred.txt"
        p.write_text("1 0.80 0.0 0.0 1.0 0.0 0.5 1.0\n", encoding="utf-8")
        preds, cids = parse_segment_predictions(str(p), IMG_W, IMG_H)
        assert len(preds) == 1
        pts, cls, conf = preds[0]
        assert cls == 1
        assert pytest.approx(conf) == 0.80
        assert len(pts) == 3
        assert cids == {1}

    def test_missing_file(self, tmp_path):
        preds, _ = parse_segment_predictions(str(tmp_path / "x.txt"), IMG_W, IMG_H)
        assert preds == []


# ── write_detect_labels ─────────────────────────────────────────────────────

class TestWriteDetectLabels:
    def test_roundtrip(self, tmp_path):
        p = str(tmp_path / "det.txt")
        boxes = [(192.0, 96.0, 448.0, 384.0, 0)]
        write_detect_labels(p, boxes, IMG_W, IMG_H)
        boxes2, cids = parse_detect_labels(p, IMG_W, IMG_H)
        assert len(boxes2) == 1
        for a, b in zip(boxes[0][:4], boxes2[0][:4]):
            assert pytest.approx(a, abs=0.1) == b
        assert boxes2[0][4] == 0
        assert cids == {0}

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
        polys2, cids = parse_segment_labels(p, IMG_W, IMG_H)
        assert len(polys2) == 1
        pts2, cls2 = polys2[0]
        assert cls2 == 1
        for (ax, ay), (bx, by) in zip(polys[0][0], pts2):
            assert pytest.approx(ax, abs=0.1) == bx
            assert pytest.approx(ay, abs=0.1) == by

    def test_empty_removes_file(self, tmp_path):
        p = tmp_path / "seg.txt"
        p.write_text("junk", encoding="utf-8")
        write_segment_labels(str(p), [], IMG_W, IMG_H)
        assert not p.exists()
