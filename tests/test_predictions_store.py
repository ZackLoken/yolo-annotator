"""Tests for yololabeler.predictions.store, prediction ids, loading and manifest."""

import hashlib

import pytest

from yololabeler.predictions.store import (
    file_hash, load_predictions, prediction_id, read_manifest, write_manifest,
)


def make_dirs(tmp_path):
    d = tmp_path / "detect"
    s = tmp_path / "segment"
    d.mkdir()
    s.mkdir()
    return d, s


# ── ids ─────────────────────────────────────────────────────────────────────

class TestIds:
    def test_file_hash_is_sha1_prefix(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_bytes(b"0 0.9 0.5 0.5 0.2 0.2\n")
        assert file_hash(p) == hashlib.sha1(b"0 0.9 0.5 0.5 0.2 0.2\n").hexdigest()[:12]

    def test_prediction_id_format(self):
        assert prediction_id("abcdef012345", 3) == "abcdef012345:3"


# ── load_predictions ────────────────────────────────────────────────────────

class TestLoadPredictions:
    def test_missing_files_are_empty(self, tmp_path):
        d, s = make_dirs(tmp_path)
        assert load_predictions(d, s, "x", 100, 100) == ([], [])

    def test_boxes_and_polygons_get_ids(self, tmp_path):
        d, s = make_dirs(tmp_path)
        (d / "x.txt").write_text("0 0.9 0.5 0.5 0.2 0.2\n1 0.4 0.1 0.1 0.1 0.1\n", encoding="utf-8")
        (s / "x.txt").write_text("2 0.7 0 0 0.5 0 0.5 0.5\n", encoding="utf-8")
        preds, rejected = load_predictions(d, s, "x", 100, 200)
        assert rejected == []
        assert [p.kind for p in preds] == ["box", "box", "polygon"]
        assert preds[0].id == f"{file_hash(d / 'x.txt')}:0"
        assert preds[1].id == f"{file_hash(d / 'x.txt')}:1"
        assert preds[2].id == f"{file_hash(s / 'x.txt')}:0"
        assert preds[0].points == ((40.0, 80.0), (60.0, 120.0))
        assert preds[2].confidence == pytest.approx(0.7)

    def test_ids_change_when_file_changes(self, tmp_path):
        d, s = make_dirs(tmp_path)
        (d / "x.txt").write_text("0 0.9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        before = load_predictions(d, s, "x", 100, 100)[0][0].id
        (d / "x.txt").write_text("0 0.8 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        after = load_predictions(d, s, "x", 100, 100)[0][0].id
        assert before != after

    def test_rejected_lines_reported_and_indexes_kept(self, tmp_path):
        d, s = make_dirs(tmp_path)
        (d / "x.txt").write_text("bad\n0 0.9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        preds, rejected = load_predictions(d, s, "x", 100, 100)
        assert rejected == [f"{d / 'x.txt'}: line 1"]
        assert preds[0].line_index == 1 and preds[0].id.endswith(":1")


# ── manifest ────────────────────────────────────────────────────────────────

class TestManifest:
    def test_absent(self, tmp_path):
        assert read_manifest(tmp_path) is None

    def test_round_trip(self, tmp_path):
        write_manifest(tmp_path, {"model": "nathan_v15"})
        assert read_manifest(tmp_path)["model"] == "nathan_v15"

    def test_corrupt_manifest_is_quarantined_and_read_as_absent(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
        assert read_manifest(tmp_path) is None
        assert not (tmp_path / "manifest.json").exists()
        assert list(tmp_path.glob("manifest.json.corrupt-*"))
