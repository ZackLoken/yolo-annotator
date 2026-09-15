"""Pair a source prediction tree with an image tree and import every folder. GUI-free."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from yololabeler.predictions.importers import (
    FORMATS, ImportResult, _source_stems, import_predictions,
)
from yololabeler.utils import is_image_file


@dataclass(frozen=True)
class Pair:
    """One source folder matched to one image folder by their shared relative path."""
    rel_path: str
    source_dir: str
    image_dir: str
    source_files: int
    images: int


@dataclass
class BatchPlan:
    """Matched pairs, the folders that paired with nothing, and colliding source stems."""
    source_root: str
    images_root: str
    fmt: str
    pairs: List[Pair] = field(default_factory=list)
    unmatched_sources: List[str] = field(default_factory=list)
    unmatched_images: List[str] = field(default_factory=list)
    stem_collisions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class PairResult:
    """One pair's import outcome; result is None on a dry run, which writes nothing."""
    pair: Pair
    result: Optional[ImportResult] = None

    def summary(self):
        """One-line human-readable summary of this pair, for the console."""
        if self.result is None:
            return (f"{self.pair.source_files} source files, "
                    f"{self.pair.images} images (dry run, nothing written).")
        return self.result.summary()


@dataclass
class BatchTotals:
    """The rollup across every pair of a batch run."""
    pairs: int = 0
    files_written: int = 0
    files_skipped: int = 0
    lines_rejected: int = 0
    files_removed: int = 0
    rotated_images: int = 0

    def summary(self):
        """One-line human-readable summary of the whole batch, for the console."""
        noun = "folder" if self.pairs == 1 else "folders"
        parts = [f"Imported predictions for {self.files_written} images "
                 f"across {self.pairs} {noun}"]
        if self.files_skipped:
            parts.append(f"{self.files_skipped} files skipped (no matching image)")
        if self.lines_rejected:
            parts.append(f"{self.lines_rejected} lines rejected")
        if self.files_removed:
            parts.append(f"{self.files_removed} existing prediction files removed "
                         "(this import had nothing for them)")
        if self.rotated_images:
            noun = "image has" if self.rotated_images == 1 else "images have"
            parts.append(f"{self.rotated_images} {noun} an EXIF rotation; "
                         "predictions are assumed to be in the rotated frame")
        return ". ".join(parts) + "."


def _rel(root, path):
    """Path relative to root, as '.' for root itself."""
    return os.path.relpath(path, root)


def _image_dirs(images_root):
    """rel_path -> image count, for every directory under images_root holding images."""
    found: Dict[str, int] = {}
    for dirpath, _dirnames, filenames in os.walk(images_root):
        count = sum(1 for name in filenames if is_image_file(name))
        if count:
            found[_rel(images_root, dirpath)] = count
    return found


def _source_dirs(source_root, fmt):
    """rel_path -> [stem], for every directory under source_root holding source files."""
    found: Dict[str, List[str]] = {}
    for dirpath, _dirnames, _filenames in os.walk(source_root):
        stems = [stem for stem, _filename in _source_stems(dirpath, fmt)]
        if stems:
            found[_rel(source_root, dirpath)] = stems
    return found


def _collisions(source_stems):
    """stem -> sorted rel_paths, for every stem found under more than one source folder."""
    seen: Dict[str, List[str]] = {}
    for rel_path, stems in source_stems.items():
        for stem in stems:
            seen.setdefault(stem, []).append(rel_path)
    return {stem: sorted(paths) for stem, paths in sorted(seen.items()) if len(paths) > 1}


def plan_pairs(source_root, images_root, fmt):
    """Pair source folders to image folders strictly by relative path, without writing.

    A source folder matches an image folder only when their paths relative to
    their own roots are identical; a leaf folder name matching somewhere else in
    the tree is not a match, and source files are never indexed by stem across
    folders. Folders that pair with nothing are reported, not resolved.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown prediction format {fmt!r}; choose one of {FORMATS}")
    source_root, images_root = str(source_root), str(images_root)
    images = _image_dirs(images_root)
    sources = _source_dirs(source_root, fmt)
    plan = BatchPlan(source_root=source_root, images_root=images_root, fmt=fmt,
                     stem_collisions=_collisions(sources))
    for rel_path in sorted(images):
        if rel_path in sources:
            plan.pairs.append(Pair(
                rel_path=rel_path,
                source_dir=os.path.join(source_root, rel_path),
                image_dir=os.path.join(images_root, rel_path),
                source_files=len(sources[rel_path]),
                images=images[rel_path]))
        else:
            plan.unmatched_images.append(rel_path)
    plan.unmatched_sources = sorted(set(sources) - set(images))
    return plan


def batch_import(plan, model_name, class_id, user, dry_run=False):
    """Run import_predictions for every matched pair; write nothing when dry_run.

    The format comes from the plan, so the counts previewed by a dry run and the
    files written by the real run can never be read with two different formats.
    """
    if plan.fmt == "bur_detect_json" and class_id is None:
        raise ValueError("bur_detect_json files carry no class; a class id is required")
    results: List[PairResult] = []
    for pair in plan.pairs:
        result = None
        if not dry_run:
            result = import_predictions(pair.source_dir, pair.image_dir, plan.fmt,
                                        model_name, class_id, user)
        results.append(PairResult(pair=pair, result=result))
    return results


def totals(results):
    """Sum every pair's ImportResult into one BatchTotals; dry-run pairs add nothing."""
    total = BatchTotals(pairs=len(results))
    for pair_result in results:
        result = pair_result.result
        if result is None:
            continue
        total.files_written += result.files_written
        total.files_skipped += len(result.files_skipped)
        total.lines_rejected += result.lines_rejected
        total.files_removed += result.files_removed
        total.rotated_images += result.rotated_images
    return total
