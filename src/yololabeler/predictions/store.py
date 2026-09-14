"""Load canonical prediction files with stable sha1+line ids, GUI-free (spec 3.2)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

from yololabeler.label_io import parse_label_file, write_json_atomic

MANIFEST_NAME = "manifest.json"
HASH_LENGTH = 12  # spec 3.2


@dataclass(frozen=True)
class Prediction:
    """One model output in image pixels with its stable id."""
    id: str
    kind: str
    points: Tuple[Tuple[float, float], ...]
    class_id: int
    confidence: float
    line_index: int


def file_hash(path):
    """Return the first HASH_LENGTH hex digits of the sha1 of path's bytes."""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:HASH_LENGTH]


def prediction_id(hash12, line_index):
    """Build a stable prediction id from a file hash and 0-based line index."""
    return f"{hash12}:{line_index}"


def load_predictions(pred_detect_dir, pred_segment_dir, stem, width, height):
    """Read an image stem's detect and segment prediction files, returning (predictions, rejected)."""
    predictions: List[Prediction] = []
    rejected: List[str] = []
    for kind, folder in (("box", pred_detect_dir), ("polygon", pred_segment_dir)):
        if not folder:
            continue
        path = os.path.join(str(folder), f"{stem}.txt")
        if not os.path.exists(path):
            continue
        digest = file_hash(path)
        parsed = parse_label_file(path, kind, width, height, with_conf=True)
        rejected.extend(f"{path}: line {n}" for n in parsed.rejected)
        for row in parsed.rows:
            predictions.append(Prediction(
                id=prediction_id(digest, row.line_index), kind=kind,
                points=tuple(row.points), class_id=row.class_id,
                confidence=row.confidence, line_index=row.line_index))
    return predictions, rejected


def read_manifest(predictions_dir):
    """Return the parsed manifest.json under predictions_dir, or None if absent."""
    path = os.path.join(str(predictions_dir), MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(predictions_dir, manifest):
    """Atomically write manifest as manifest.json under predictions_dir."""
    write_json_atomic(os.path.join(str(predictions_dir), MANIFEST_NAME), manifest)
