"""Tests for yololabeler.predictions.cli, the headless yololabeler-import entry
point.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from yololabeler.predictions.cli import main

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def tree(tmp_path):
    """Source tree and image tree pairing on one mission, with one unmatched
    folder each.
    """
    src, img = tmp_path / "preds", tmp_path / "images"
    for root, rel in ((img, "Farm/North/M1"), (img, "Farm/North/M0")):
        (root / rel).mkdir(parents=True)
        Image.new("RGB", (100, 200)).save(root / rel / "a.jpg")
    for rel in ("Farm/North/M1", "Farm/South/M9"):
        (src / rel).mkdir(parents=True)
        (src / rel / "a.json").write_text(
            json.dumps({"boxes": [[10, 20, 30, 60]], "scores": [0.875]}),
            encoding="utf-8",
        )
    return src, img


def run(args):
    return main([str(a) for a in args])


def files_under(root):
    return sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )


# ── import chain ────────────────────────────────────────────────────────────


class TestHeadless:
    def test_no_tk_in_the_import_chain(self):
        """A fresh interpreter importing the CLI must load no Tk or
        CustomTkinter.
        """
        env = dict(os.environ, PYTHONPATH=str(SRC))
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, yololabeler.predictions.cli;"
                "print([m for m in sys.modules if m.split('.')[0] in"
                " ('tkinter', '_tkinter', 'customtkinter')])",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "[]", proc.stdout


# ── root overlap rejection ──────────────────────────────────────────────────


class TestRootOverlap:
    def test_identical_roots_rejected(self, tree):
        _src, img = tree
        with pytest.raises(SystemExit) as exc:
            run(
                [
                    img,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                ]
            )
        assert exc.value.code == 2

    def test_source_nested_in_images_rejected(self, tree):
        _src, img = tree
        nested = img / "preds"
        nested.mkdir()
        with pytest.raises(SystemExit) as exc:
            run(
                [
                    nested,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                ]
            )
        assert exc.value.code == 2

    def test_images_nested_in_source_rejected(self, tmp_path):
        src = tmp_path / "preds"
        nested = src / "images"
        nested.mkdir(parents=True)
        with pytest.raises(SystemExit) as exc:
            run(
                [
                    src,
                    nested,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                ]
            )
        assert exc.value.code == 2

    def test_sibling_roots_accepted(self, tree):
        src, img = tree
        assert (
            run(
                [
                    src,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                ]
            )
            == 0
        )

    def test_missing_root_rejected(self, tmp_path, tree):
        _src, img = tree
        with pytest.raises(SystemExit) as exc:
            run(
                [
                    tmp_path / "nope",
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                ]
            )
        assert exc.value.code == 2

    def test_bur_detect_json_requires_class_id(self, tree):
        src, img = tree
        with pytest.raises(SystemExit) as exc:
            run([src, img, "--format", "bur_detect_json", "--model", "m"])
        assert exc.value.code == 2


# ── dry run ─────────────────────────────────────────────────────────────────


class TestDryRun:
    def test_is_the_default_and_writes_nothing(self, tree, capsys):
        src, img = tree
        before_src, before_img = files_under(src), files_under(img)
        assert (
            run(
                [
                    src,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                ]
            )
            == 0
        )
        assert (
            files_under(src) == before_src and files_under(img) == before_img
        )
        assert "Dry run: nothing written" in capsys.readouterr().out

    def test_prints_pairs_and_both_unmatched_lists(self, tree, capsys):
        src, img = tree
        run(
            [
                src,
                img,
                "--format",
                "bur_detect_json",
                "--model",
                "m",
                "--class-id",
                0,
            ]
        )
        out = capsys.readouterr().out
        assert "Matched folder pairs (1):" in out
        assert (
            "Farm/North/M1" in out
            and "Farm/North/M0" in out
            and "Farm/South/M9" in out
        )
        assert "1 source files" in out and "1 images" in out

    def test_prints_stem_collisions(self, tmp_path, capsys):
        src, img = tmp_path / "preds", tmp_path / "images"
        for rel in ("Farm/North/M1", "Farm/South/M1"):
            (img / rel).mkdir(parents=True)
            Image.new("RGB", (100, 200)).save(img / rel / "DJI_0001.jpg")
            (src / rel).mkdir(parents=True)
            (src / rel / "DJI_0001.json").write_text(
                json.dumps({"boxes": [], "scores": []}), encoding="utf-8"
            )
        run(
            [
                src,
                img,
                "--format",
                "bur_detect_json",
                "--model",
                "m",
                "--class-id",
                0,
            ]
        )
        out = capsys.readouterr().out
        assert (
            "Source stems found under more than one source folder (1):" in out
        )
        assert "DJI_0001: Farm/North/M1, Farm/South/M1" in out

    def test_explicit_dry_run_flag_also_writes_nothing(self, tree):
        src, img = tree
        before = files_under(img)
        assert (
            run(
                [
                    src,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                    "--dry-run",
                ]
            )
            == 0
        )
        assert files_under(img) == before

    def test_dry_run_and_write_are_mutually_exclusive(self, tree):
        src, img = tree
        with pytest.raises(SystemExit) as exc:
            run(
                [
                    src,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                    "--dry-run",
                    "--write",
                ]
            )
        assert exc.value.code == 2

    def test_no_pairs_is_a_nonzero_exit(self, tmp_path, capsys):
        src, img = tmp_path / "preds", tmp_path / "images"
        src.mkdir()
        (img / "Farm/North/M1").mkdir(parents=True)
        Image.new("RGB", (100, 200)).save(img / "Farm/North/M1" / "a.jpg")
        assert (
            run(
                [
                    src,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "m",
                    "--class-id",
                    0,
                ]
            )
            == 1
        )
        assert "No folder pairs matched" in capsys.readouterr().out


# ── write ───────────────────────────────────────────────────────────────────


class TestWrite:
    def test_writes_matched_pairs_only(self, tree, capsys):
        src, img = tree
        assert (
            run(
                [
                    src,
                    img,
                    "--format",
                    "bur_detect_json",
                    "--model",
                    "nathan_v15",
                    "--class-id",
                    0,
                    "--user",
                    "zack",
                    "--write",
                ]
            )
            == 0
        )
        assert (
            img / "Farm/North/M1" / "predictions" / "detect" / "a.txt"
        ).read_text(encoding="utf-8").splitlines() == [
            "0 0.875000 0.200000 0.200000 0.200000 0.200000"
        ]
        assert not (img / "Farm/North/M0" / "predictions").exists()
        out = capsys.readouterr().out
        assert "Farm/North/M1: Imported predictions for 1 images" in out
        assert "across 1 folder." in out
