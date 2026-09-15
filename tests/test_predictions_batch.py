"""Tests for yololabeler.predictions.batch, the tree-to-tree prediction importer."""

import json

import pytest
from PIL import Image

from yololabeler.predictions.batch import (
    BatchTotals, batch_import, plan_pairs, totals,
)
from yololabeler.predictions.importers import import_predictions
from yololabeler.predictions.store import read_manifest

# Farm/Field/Mission-shaped tree: two fields, two missions each, one mission per
# side that the other side has no folder for.
MISSIONS = ("Farm/North/M1", "Farm/North/M2", "Farm/South/M1", "Farm/South/M2")
SOURCE_ONLY = "Farm/South/M9"
IMAGES_ONLY = "Farm/North/M0"


def make_image(path, stem):
    path.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 200)).save(path / f"{stem}.jpg")


def make_json(path, stem, boxes=((10, 20, 30, 60),), scores=(0.875,)):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{stem}.json").write_text(
        json.dumps({"boxes": [list(b) for b in boxes], "scores": list(scores)}),
        encoding="utf-8")


def stem_for(rel_path):
    """A stem unique to one mission, so a mispairing shows up as a missing file."""
    return "img_" + rel_path.replace("/", "_")


@pytest.fixture
def tree(tmp_path):
    """Source tree and image tree that pair on MISSIONS and differ on one folder each."""
    src, img = tmp_path / "preds", tmp_path / "images"
    for rel in MISSIONS:
        make_image(img / rel, stem_for(rel))
        make_json(src / rel, stem_for(rel))
    make_json(src / SOURCE_ONLY, stem_for(SOURCE_ONLY))
    make_image(img / IMAGES_ONLY, stem_for(IMAGES_ONLY))
    return src, img


def files_under(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


# ── plan_pairs ──────────────────────────────────────────────────────────────

class TestPlanPairs:
    def test_pairs_by_relative_path_through_a_multi_level_tree(self, tree):
        src, img = tree
        plan = plan_pairs(src, img, "bur_detect_json")
        assert [p.rel_path.replace("\\", "/") for p in plan.pairs] == sorted(MISSIONS)
        for pair in plan.pairs:
            assert pair.source_files == 1 and pair.images == 1

    def test_same_leaf_name_under_a_different_parent_is_not_a_match(self, tmp_path):
        src, img = tmp_path / "preds", tmp_path / "images"
        make_image(img / "Farm/North/M1", "a")
        make_json(src / "Farm/South/M1", "a")  # same leaf name, different branch
        plan = plan_pairs(src, img, "bur_detect_json")
        assert plan.pairs == []
        assert [p.replace("\\", "/") for p in plan.unmatched_images] == ["Farm/North/M1"]
        assert [p.replace("\\", "/") for p in plan.unmatched_sources] == ["Farm/South/M1"]

    def test_unmatched_source_folders_are_reported(self, tree):
        src, img = tree
        plan = plan_pairs(src, img, "bur_detect_json")
        assert [p.replace("\\", "/") for p in plan.unmatched_sources] == [SOURCE_ONLY]

    def test_unmatched_image_folders_are_reported(self, tree):
        src, img = tree
        plan = plan_pairs(src, img, "bur_detect_json")
        assert [p.replace("\\", "/") for p in plan.unmatched_images] == [IMAGES_ONLY]

    def test_cross_folder_stem_collisions_are_reported(self, tmp_path):
        src, img = tmp_path / "preds", tmp_path / "images"
        for rel in ("Farm/North/M1", "Farm/South/M1"):
            make_image(img / rel, "DJI_0001")
            make_json(src / rel, "DJI_0001")
        plan = plan_pairs(src, img, "bur_detect_json")
        assert len(plan.pairs) == 2
        collisions = {stem: [p.replace("\\", "/") for p in paths]
                      for stem, paths in plan.stem_collisions.items()}
        assert collisions == {"DJI_0001": ["Farm/North/M1", "Farm/South/M1"]}

    def test_unique_stems_report_no_collisions(self, tree):
        src, img = tree
        assert plan_pairs(src, img, "bur_detect_json").stem_collisions == {}

    def test_empty_source_folder_does_not_pair(self, tmp_path):
        src, img = tmp_path / "preds", tmp_path / "images"
        make_image(img / "Farm/North/M1", "a")
        (src / "Farm/North/M1").mkdir(parents=True)
        plan = plan_pairs(src, img, "bur_detect_json")
        assert plan.pairs == [] and len(plan.unmatched_images) == 1

    def test_yololabeler_format_finds_the_folder_holding_detect_and_segment(self, tmp_path):
        src, img = tmp_path / "preds", tmp_path / "images"
        make_image(img / "Farm/North/M1", "a")
        (src / "Farm/North/M1" / "detect").mkdir(parents=True)
        (src / "Farm/North/M1" / "detect" / "a.txt").write_text(
            "0 0.9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        plan = plan_pairs(src, img, "yololabeler")
        assert [p.rel_path.replace("\\", "/") for p in plan.pairs] == ["Farm/North/M1"]
        assert plan.unmatched_sources == []

    def test_unknown_format(self, tree):
        src, img = tree
        with pytest.raises(ValueError, match="format"):
            plan_pairs(src, img, "nope")


# ── batch_import ────────────────────────────────────────────────────────────

class TestBatchImportDryRun:
    def test_writes_nothing(self, tree):
        src, img = tree
        before_src, before_img = files_under(src), files_under(img)
        batch_import(plan_pairs(src, img, "bur_detect_json"), "m", 0, "z", dry_run=True)
        assert files_under(src) == before_src
        assert files_under(img) == before_img

    def test_reports_accurate_counts(self, tree):
        src, img = tree
        results = batch_import(plan_pairs(src, img, "bur_detect_json"),
                               "m", 0, "z", dry_run=True)
        assert len(results) == len(MISSIONS)
        assert all(r.result is None for r in results)
        assert all(r.pair.source_files == 1 and r.pair.images == 1 for r in results)
        assert "dry run" in results[0].summary()

    def test_requires_class_id(self, tree):
        src, img = tree
        with pytest.raises(ValueError, match="class id"):
            batch_import(plan_pairs(src, img, "bur_detect_json"), "m", None, "z",
                         dry_run=True)


class TestBatchImportWrite:
    def test_matches_per_pair_import_predictions(self, tree, tmp_path):
        src, img = tree
        results = batch_import(plan_pairs(src, img, "bur_detect_json"), "m", 0, "z")
        assert len(results) == len(MISSIONS)
        for rel in MISSIONS:
            stem = stem_for(rel)
            written = (img / rel / "predictions" / "detect" / f"{stem}.txt")
            assert written.exists()
            # The same source folder imported on its own, into a fresh copy.
            solo = tmp_path / "solo" / rel
            make_image(solo, stem)
            import_predictions(src / rel, solo, "bur_detect_json", "m", 0, "z")
            assert written.read_text(encoding="utf-8") == \
                (solo / "predictions" / "detect" / f"{stem}.txt").read_text(encoding="utf-8")

    def test_does_not_write_into_unmatched_folders(self, tree):
        src, img = tree
        batch_import(plan_pairs(src, img, "bur_detect_json"), "m", 0, "z")
        assert not (img / IMAGES_ONLY / "predictions").exists()
        assert not (src / SOURCE_ONLY / "predictions").exists()

    def test_manifest_records_the_source_dir_of_that_pair(self, tree):
        src, img = tree
        batch_import(plan_pairs(src, img, "bur_detect_json"), "nathan_v15", 0, "zack")
        for rel in MISSIONS:
            manifest = read_manifest(img / rel / "predictions")
            assert manifest["source_dir"] == str(src / rel)
            assert manifest["model"] == "nathan_v15"
            assert manifest["files"] == 1

    def test_rollup_totals(self, tree):
        src, img = tree
        results = batch_import(plan_pairs(src, img, "bur_detect_json"), "m", 0, "z")
        total = totals(results)
        assert total.pairs == len(MISSIONS)
        assert total.files_written == len(MISSIONS)
        assert total.files_skipped == 0 and total.lines_rejected == 0
        assert "across 4 folders" in total.summary()

    def test_rollup_counts_skipped_and_rejected(self, tmp_path):
        src, img = tmp_path / "preds", tmp_path / "images"
        make_image(img / "Farm/North/M1", "a")
        make_json(src / "Farm/North/M1", "a", boxes=[(10, 20, 30, 60), (0, 0, 5, 5)],
                  scores=[0.9])
        make_json(src / "Farm/North/M1", "no_such_image")
        results = batch_import(plan_pairs(src, img, "bur_detect_json"), "m", 0, "z")
        total = totals(results)
        assert total.files_skipped == 1 and total.lines_rejected == 1
        assert "1 files skipped" in total.summary() and "1 lines rejected" in total.summary()


class TestTotals:
    def test_dry_run_results_add_nothing(self, tree):
        src, img = tree
        results = batch_import(plan_pairs(src, img, "bur_detect_json"),
                               "m", 0, "z", dry_run=True)
        total = totals(results)
        assert total.pairs == len(MISSIONS) and total.files_written == 0

    def test_empty_summary(self):
        assert BatchTotals().summary() == \
            "Imported predictions for 0 images across 0 folders."
