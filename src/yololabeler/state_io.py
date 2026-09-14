"""Read and write state/annotation_stats.json, GUI-free.

A JSON file that fails to parse is renamed aside rather than replaced, so a
crash mid-write or a hand edit never silently costs the history (spec 7.3).
"""

from __future__ import annotations

import datetime
import json
import os

from yololabeler.label_io import write_json_atomic


def read_json_or_quarantine(path):
    """Return (data, None); (None, None) if missing; (None, moved_path) if corrupt."""
    path = str(path)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except ValueError:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        moved = f"{path}.corrupt-{stamp}"
        os.replace(path, moved)
        return None, moved


class AnnotationStats:
    """Image status, blind flags, completion records and session history."""

    def __init__(self, data=None):
        self.data = data or {}
        self.data.setdefault("sessions", [])
        self.data.setdefault("image_status", {})
        self.data.setdefault("blind", [])
        self.data.setdefault("completion", {})
        legacy = self.data.pop("images", None)
        if legacy:
            for name, entry in legacy.items():
                if entry.get("status") == "complete":
                    self.data["image_status"].setdefault(name, "complete")

    @classmethod
    def load(cls, path):
        """Read stats from path, quarantining a corrupt file, and return (instance, moved_path)."""
        data, moved = read_json_or_quarantine(path)
        return cls(data), moved

    def save(self, path):
        """Write the current stats to path atomically."""
        write_json_atomic(path, self.data)

    @property
    def sessions(self):
        """Return the list of recorded session entries."""
        return self.data["sessions"]

    def image_status(self, name):
        """Return the recorded status for name, defaulting to 'unannotated'."""
        return self.data["image_status"].get(name, "unannotated")

    def set_image_status(self, name, status):
        """Set the recorded status for name."""
        self.data["image_status"][name] = status

    def is_blind(self, name):
        """Return whether name is flagged for blind review."""
        return name in self.data["blind"]

    def set_blind(self, name, flag):
        """Set or clear the blind-review flag for name."""
        blind = self.data["blind"]
        if flag and name not in blind:
            blind.append(name)
        if not flag and name in blind:
            blind.remove(name)

    def completion(self, name):
        """Return the completion record for name, or None if not completed."""
        return self.data["completion"].get(name)

    def set_completion(self, name, by, blind, annotation_count, model):
        """Record a completion entry for name with author, timestamp, blind flag, count and model."""
        self.data["completion"][name] = {
            "by": by, "at": datetime.datetime.now().isoformat(timespec="seconds"),
            "blind": bool(blind), "annotation_count": int(annotation_count),
            "model": model}

    def clear_completion(self, name):
        """Remove the completion record for name, if any."""
        self.data["completion"].pop(name, None)

    def pop_legacy_authors(self, name):
        """Remove and return this image's old parallel author lists, if any."""
        authors = self.data.get("annotation_authors")
        if not authors or name not in authors:
            return None
        entry = authors.pop(name)
        if not authors:
            del self.data["annotation_authors"]
        return list(entry.get("boxes", [])), list(entry.get("polygons", []))
