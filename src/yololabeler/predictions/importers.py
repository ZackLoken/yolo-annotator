"""Convert external prediction files into the canonical on-disk layout (spec 4.2)."""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from yololabeler.label_io import (
    _write_label_lines, format_detect_line, format_segment_line, parse_label_file,
)
from yololabeler.predictions.store import write_manifest
from yololabeler.utils import oriented_size

FORMATS = ("yololabeler", "ultralytics_txt", "bur_detect_json")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class ImportResult:
    """What an import wrote, skipped and rejected, for the banner."""
    files_written: int = 0
    files_skipped: List[str] = field(default_factory=list)
    lines_rejected: int = 0
    rotated_images: int = 0

    def summary(self):
        """One-line human-readable summary of the import, for the status banner."""
        parts = [f"Imported predictions for {self.files_written} images"]
        if self.files_skipped:
            parts.append(f"{len(self.files_skipped)} files skipped (no matching image)")
        if self.lines_rejected:
            parts.append(f"{self.lines_rejected} lines rejected")
        if self.rotated_images:
            noun = "image has" if self.rotated_images == 1 else "images have"
            parts.append(f"{self.rotated_images} {noun} an EXIF rotation; "
                         "predictions are assumed to be in the rotated frame")
        return ". ".join(parts) + "."


def _image_index(image_folder):
    """stem -> (width, height, orientation) for every image in the folder."""
    index: Dict[str, Tuple[int, int, int]] = {}
    for name in sorted(os.listdir(image_folder)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            index[os.path.splitext(name)[0]] = oriented_size(os.path.join(image_folder, name))
    return index


def _detect_line(class_id, conf, x1, y1, x2, y2, w, h):
    """Format one canonical detect prediction line with confidence in column two."""
    return format_detect_line(x1, y1, x2, y2, class_id, w, h).replace(
        f"{class_id} ", f"{class_id} {conf:.6f} ", 1)


def _segment_line(class_id, conf, points, w, h):
    """Format one canonical segment prediction line with confidence in column two."""
    return format_segment_line(points, class_id, w, h).replace(
        f"{class_id} ", f"{class_id} {conf:.6f} ", 1)


def _convert_bur_json(path, class_id, w, h):
    """Convert one bur_detect_json file's pixel boxes into canonical detect lines."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for (x1, y1, x2, y2), score in zip(data.get("boxes", []), data.get("scores", [])):
        lines.append(_detect_line(class_id, float(score), x1, y1, x2, y2, w, h))
    return lines, [], 0


def _convert_ultralytics(path, w, h):
    """Convert one Ultralytics save_txt(save_conf=True) file, moving confidence to column two."""
    detect, segment, rejected = [], [], 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                rejected += 1
                continue
            if len(vals) == 5:
                cx, cy, bw, bh, conf = vals
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                detect.append(_detect_line(class_id, conf, x1, y1, x2, y2, w, h))
            elif len(vals) >= 7 and len(vals) % 2 == 1:
                conf = vals[-1]
                pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, len(vals) - 1, 2)]
                segment.append(_segment_line(class_id, conf, pts, w, h))
            else:
                rejected += 1
    return detect, segment, rejected


def _convert_yololabeler(source_dir, stem, w, h):
    """Re-format one stem's canonical-layout detect and segment prediction files."""
    out = {}
    rejected = 0
    for kind, sub in (("box", "detect"), ("polygon", "segment")):
        path = os.path.join(source_dir, sub, f"{stem}.txt")
        parsed = parse_label_file(path, kind, w, h, with_conf=True)
        rejected += len(parsed.rejected)
        if kind == "box":
            out[sub] = [_detect_line(r.class_id, r.confidence, *r.points[0], *r.points[1], w, h)
                        for r in parsed.rows]
        else:
            out[sub] = [_segment_line(r.class_id, r.confidence, r.points, w, h)
                        for r in parsed.rows]
    return out["detect"], out["segment"], rejected


def _source_stems(source_dir, fmt):
    """List (stem, filename) pairs of source files to convert, for the given format."""
    if fmt == "yololabeler":
        stems = set()
        for sub in ("detect", "segment"):
            folder = os.path.join(source_dir, sub)
            if os.path.isdir(folder):
                stems.update(os.path.splitext(n)[0] for n in os.listdir(folder)
                             if n.endswith(".txt"))
        return sorted((s, s) for s in stems)
    ext = ".json" if fmt == "bur_detect_json" else ".txt"
    return sorted((os.path.splitext(n)[0], n) for n in os.listdir(source_dir)
                  if n.endswith(ext))


def import_predictions(source_dir, image_folder, fmt, model_name, class_id, user):
    """Convert source files into image_folder/predictions and write the manifest."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown prediction format {fmt!r}; choose one of {FORMATS}")
    if fmt == "bur_detect_json" and class_id is None:
        raise ValueError("bur_detect_json files carry no class; a class id is required")
    source_dir, image_folder = str(source_dir), str(image_folder)
    images = _image_index(image_folder)
    result = ImportResult(rotated_images=sum(1 for _, _, o in images.values() if o != 1))
    converted: Dict[str, Tuple[List[str], List[str]]] = {}
    for stem, filename in _source_stems(source_dir, fmt):
        if stem not in images:
            result.files_skipped.append(filename)
            continue
        w, h, _ = images[stem]
        if fmt == "bur_detect_json":
            detect, segment, rejected = _convert_bur_json(
                os.path.join(source_dir, filename), class_id, w, h)
        elif fmt == "ultralytics_txt":
            detect, segment, rejected = _convert_ultralytics(
                os.path.join(source_dir, filename), w, h)
        else:
            detect, segment, rejected = _convert_yololabeler(source_dir, stem, w, h)
        result.lines_rejected += rejected
        converted[stem] = (detect, segment)
    detect_dir = os.path.join(image_folder, "predictions", "detect")
    segment_dir = os.path.join(image_folder, "predictions", "segment")
    os.makedirs(detect_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)
    for stem, (detect, segment) in converted.items():
        _write_label_lines(os.path.join(detect_dir, f"{stem}.txt"), [l + "\n" for l in detect])
        _write_label_lines(os.path.join(segment_dir, f"{stem}.txt"), [l + "\n" for l in segment])
        result.files_written += 1
    write_manifest(os.path.join(image_folder, "predictions"), {
        "model": model_name, "source_format": fmt,
        "imported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "imported_by": user, "class_id_default": class_id,
        "files": result.files_written})
    return result
