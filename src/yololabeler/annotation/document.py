"""Per-image annotation document with provenance, GUI-free.

Label files stay the geometry of record. A sidecar JSON next to them carries id,
author, creation time and provenance per annotation, joined to label lines by
the exact formatted line text (spec section 6.2).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from yololabeler.label_io import (
    format_detect_line, format_segment_line, parse_label_file,
    write_detect_labels, write_json_atomic, write_segment_labels,
)

Point = Tuple[float, float]
SOURCES = ("drawn", "accepted", "unknown")


@dataclass(frozen=True)
class Annotation:
    """One box or polygon in image pixels plus who made it and where it came from."""
    id: str
    kind: str
    points: Tuple[Point, ...]
    class_id: int
    author: str = ""
    created: str = ""
    source: str = "drawn"
    prediction_id: Optional[str] = None
    confidence: Optional[float] = None


def new_annotation(kind, points, class_id, author, source="drawn",
                   prediction_id=None, confidence=None):
    """Build an Annotation with a fresh uuid4 id and the current local time."""
    return Annotation(
        id=str(uuid.uuid4()), kind=kind,
        points=tuple((float(x), float(y)) for x, y in points),
        class_id=int(class_id), author=author,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        source=source, prediction_id=prediction_id, confidence=confidence)


class Document:
    """All annotations of one image, in insertion order."""

    def __init__(self, image_name, width, height, annotations=()):
        self.image_name = image_name
        self.width = width
        self.height = height
        self.annotations: List[Annotation] = list(annotations)

    def add(self, annotation):
        """Append an annotation to the end of the document."""
        self.annotations.append(annotation)

    def _index(self, ann_id):
        for i, a in enumerate(self.annotations):
            if a.id == ann_id:
                return i
        raise KeyError(ann_id)

    def get(self, ann_id):
        """Return the annotation with the given id, or raise KeyError."""
        return self.annotations[self._index(ann_id)]

    def remove(self, ann_id):
        """Remove and return the annotation with the given id."""
        return self.annotations.pop(self._index(ann_id))

    def replace(self, ann_id, **changes):
        """Return a copy of the annotation with changes applied, stored in place."""
        i = self._index(ann_id)
        if "points" in changes:
            changes["points"] = tuple((float(x), float(y)) for x, y in changes["points"])
        self.annotations[i] = dataclasses.replace(self.annotations[i], **changes)
        return self.annotations[i]

    def boxes(self):
        """Return the box annotations, in insertion order."""
        return [a for a in self.annotations if a.kind == "box"]

    def polygons(self):
        """Return the polygon annotations, in insertion order."""
        return [a for a in self.annotations if a.kind == "polygon"]

    def snapshot(self):
        """Return an immutable copy of the current annotations for undo/redo."""
        return tuple(self.annotations)

    def restore(self, snap):
        """Replace the annotations with a previously taken snapshot."""
        self.annotations = list(snap)

    def line_for(self, annotation):
        """The exact label line this annotation writes; also the sidecar join key."""
        if annotation.kind == "box":
            (x1, y1), (x2, y2) = annotation.points
            return format_detect_line(x1, y1, x2, y2, annotation.class_id,
                                      self.width, self.height)
        return format_segment_line(annotation.points, annotation.class_id,
                                   self.width, self.height)

    def label_lines(self):
        """Return the (detect, segment) label lines for all annotations."""
        detect = [self.line_for(a) for a in self.boxes()]
        segment = [self.line_for(a) for a in self.polygons()]
        return detect, segment


def _sidecar_record(doc, a):
    return {"id": a.id, "kind": a.kind, "class_id": a.class_id,
            "line": doc.line_for(a), "author": a.author, "created": a.created,
            "source": a.source, "prediction_id": a.prediction_id,
            "confidence": a.confidence}


def save_document(doc, detect_path, segment_path, sidecar_path):
    """Write label files and sidecar; an empty document removes all three."""
    write_detect_labels(str(detect_path), [(*a.points[0], *a.points[1], a.class_id)
                                           for a in doc.boxes()], doc.width, doc.height)
    write_segment_labels(str(segment_path), [(a.points, a.class_id)
                                             for a in doc.polygons()], doc.width, doc.height)
    if not doc.annotations:
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        return
    write_json_atomic(sidecar_path, {
        "image": doc.image_name, "width": doc.width, "height": doc.height,
        "annotations": [_sidecar_record(doc, a) for a in doc.annotations]})


def _read_sidecar(sidecar_path):
    if not os.path.exists(sidecar_path):
        return {}
    with open(sidecar_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {(r["kind"], r["line"]): r for r in data.get("annotations", [])}


def _from_row(kind, row, record, author):
    if record is None:
        return Annotation(id=str(uuid.uuid4()), kind=kind, points=tuple(row.points),
                          class_id=row.class_id, author=author, created="",
                          source="unknown")
    return Annotation(id=record["id"], kind=kind, points=tuple(row.points),
                      class_id=row.class_id, author=record.get("author", ""),
                      created=record.get("created", ""),
                      source=record.get("source", "unknown"),
                      prediction_id=record.get("prediction_id"),
                      confidence=record.get("confidence"))


def _canonical(row, kind, width, height):
    """Re-format a parsed row so hand-edited spacing still joins to its record."""
    if kind == "box":
        (x1, y1), (x2, y2) = row.points
        return format_detect_line(x1, y1, x2, y2, row.class_id, width, height)
    return format_segment_line(row.points, row.class_id, width, height)


def load_document(image_name, width, height, detect_path, segment_path,
                  sidecar_path, legacy_authors=None):
    """Join label lines with the sidecar. Returns (document, rejected-line messages).

    legacy_authors is an optional (box_authors, polygon_authors) pair from the old
    annotation_stats.json layout, applied by position only to lines that have no
    sidecar record.
    """
    records = _read_sidecar(sidecar_path)
    box_authors, poly_authors = legacy_authors or ([], [])
    doc = Document(image_name, width, height)
    rejected: List[str] = []
    for kind, path, authors in (("box", detect_path, box_authors),
                                ("polygon", segment_path, poly_authors)):
        parsed = parse_label_file(path, kind, width, height)
        rejected.extend(f"{path}: line {n}" for n in parsed.rejected)
        for pos, row in enumerate(parsed.rows):
            key = (kind, _canonical(row, kind, width, height))
            author = authors[pos] if pos < len(authors) else ""
            doc.add(_from_row(kind, row, records.get(key), author))
    return doc, rejected
