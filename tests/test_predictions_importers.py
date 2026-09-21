"""Tests for yololabeler.predictions.importers, the three prediction
converters.
"""

import json

import pytest
from PIL import Image

from yololabeler.predictions.importers import FORMATS, import_predictions
from yololabeler.predictions.store import read_manifest


@pytest.fixture
def folder(tmp_path):
    """Image folder with two 100x200 images, the second EXIF-rotated."""
    img = tmp_path / "images"
    img.mkdir()
    Image.new("RGB", (100, 200)).save(img / "a.jpg")
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (200, 100)).save(img / "b.jpg", exif=exif)
    return img


def read(path):
    return path.read_text(encoding="utf-8").splitlines()


# ── bur_detect_json ─────────────────────────────────────────────────────────


class TestBurDetectJson:
    def test_converts_pixels_to_canonical(self, tmp_path, folder):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.json").write_text(
            json.dumps({"boxes": [[10, 20, 30, 60]], "scores": [0.875]}),
            encoding="utf-8",
        )
        result = import_predictions(
            src, folder, "bur_detect_json", "nathan_v15", 0, "zack"
        )
        assert result.files_written == 1
        assert read(folder / "predictions" / "detect" / "a.txt") == [
            "0 0.875000 0.200000 0.200000 0.200000 0.200000"
        ]
        manifest = read_manifest(folder / "predictions")
        assert manifest["model"] == "nathan_v15"
        assert manifest["source_format"] == "bur_detect_json"
        assert manifest["imported_by"] == "zack"
        assert manifest["class_id_default"] == 0
        assert manifest["min_conf"] == pytest.approx(0.875)

    def test_unpaired_boxes_are_rejected_not_truncated(self, tmp_path, folder):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.json").write_text(
            json.dumps(
                {
                    "boxes": [[10, 20, 30, 60], [0, 0, 10, 10]],
                    "scores": [0.875],
                }
            ),
            encoding="utf-8",
        )
        result = import_predictions(
            src, folder, "bur_detect_json", "m", 0, "z"
        )
        assert result.lines_rejected == 1
        assert len(read(folder / "predictions" / "detect" / "a.txt")) == 1

    def test_removed_existing_predictions_are_reported(self, tmp_path, folder):
        existing = folder / "predictions" / "segment"
        existing.mkdir(parents=True)
        (existing / "a.txt").write_text(
            "0 0.9 0 0 0.5 0 0.5 0.5\n", encoding="utf-8"
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.json").write_text(
            json.dumps({"boxes": [[10, 20, 30, 60]], "scores": [0.875]}),
            encoding="utf-8",
        )
        result = import_predictions(
            src, folder, "bur_detect_json", "m", 0, "z"
        )
        assert result.files_removed == 1 and not (existing / "a.txt").exists()
        assert "1 existing prediction files removed" in result.summary()

    def test_requires_class_id(self, tmp_path, folder):
        with pytest.raises(ValueError, match="class id"):
            import_predictions(
                tmp_path, folder, "bur_detect_json", "m", None, "z"
            )

    def test_counts_rotated_images(self, tmp_path, folder):
        src = tmp_path / "src"
        src.mkdir()
        (src / "b.json").write_text(
            json.dumps({"boxes": [], "scores": []}), encoding="utf-8"
        )
        result = import_predictions(
            src, folder, "bur_detect_json", "m", 0, "z"
        )
        assert result.rotated_images == 1
        assert "1 image has an EXIF rotation" in result.summary()

    def test_skips_files_without_image(self, tmp_path, folder):
        src = tmp_path / "src"
        src.mkdir()
        (src / "zzz.json").write_text(
            json.dumps({"boxes": [], "scores": []}), encoding="utf-8"
        )
        result = import_predictions(
            src, folder, "bur_detect_json", "m", 0, "z"
        )
        assert result.files_written == 0 and result.files_skipped == [
            "zzz.json"
        ]

    def test_empty_result_does_not_count_as_written(self, tmp_path, folder):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.json").write_text(
            json.dumps({"boxes": [], "scores": []}), encoding="utf-8"
        )
        result = import_predictions(
            src, folder, "bur_detect_json", "m", 0, "z"
        )
        assert result.files_written == 0
        assert not (folder / "predictions" / "detect" / "a.txt").exists()


# ── ultralytics_txt ─────────────────────────────────────────────────────────


class TestUltralyticsTxt:
    def test_moves_confidence_to_column_two(self, tmp_path, folder):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.9\n2 0 0 0.5 0 0.5 0.5 0.7\n",
            encoding="utf-8",
        )
        result = import_predictions(
            src, folder, "ultralytics_txt", "m", None, "z"
        )
        assert result.files_written == 1 and result.lines_rejected == 0
        assert read(folder / "predictions" / "detect" / "a.txt") == [
            "0 0.900000 0.500000 0.500000 0.200000 0.200000"
        ]
        assert read(folder / "predictions" / "segment" / "a.txt") == [
            "2 0.700000 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000"
        ]
        manifest = read_manifest(folder / "predictions")
        assert manifest["min_conf"] == pytest.approx(0.7)

    def test_line_without_confidence_is_rejected(self, tmp_path, folder):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = import_predictions(
            src, folder, "ultralytics_txt", "m", None, "z"
        )
        assert result.lines_rejected == 1
        assert not (folder / "predictions" / "detect" / "a.txt").exists()


# ── yololabeler ─────────────────────────────────────────────────────────────


class TestYololabeler:
    def test_copies_after_validation(self, tmp_path, folder):
        src = tmp_path / "src"
        (src / "detect").mkdir(parents=True)
        (src / "detect" / "a.txt").write_text(
            "0 0.9 0.5 0.5 0.2 0.2\nbad\n", encoding="utf-8"
        )
        result = import_predictions(src, folder, "yololabeler", "m", None, "z")
        assert result.files_written == 1 and result.lines_rejected == 1
        assert read(folder / "predictions" / "detect" / "a.txt") == [
            "0 0.900000 0.500000 0.500000 0.200000 0.200000"
        ]

    def test_unknown_format(self, tmp_path, folder):
        with pytest.raises(ValueError, match="format"):
            import_predictions(tmp_path, folder, "nope", "m", None, "z")

    def test_formats_tuple(self):
        assert FORMATS == ("yololabeler", "ultralytics_txt", "bur_detect_json")
