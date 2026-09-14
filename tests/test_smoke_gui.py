"""Headed smoke test: build the real window and drive the canvas; skipped without a display."""

import pytest
from PIL import Image

pytest.importorskip("customtkinter")
import customtkinter as ctk  # noqa: E402

from yololabeler.gui import YoloLabeler  # noqa: E402


def click_at(tab, ix, iy):
    """A synthetic click event on the canvas point showing image pixel (ix, iy)."""
    cx, cy = tab.image_to_canvas(ix, iy)
    return type("E", (), {"x": int(cx), "y": int(cy)})()


@pytest.fixture
def folder(tmp_path):
    """An image folder with one labelled image, one unlabelled, and predictions."""
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (640, 480), "gray").save(tmp_path / name)
    d = tmp_path / "labels" / "detect"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    p = tmp_path / "predictions" / "detect"
    p.mkdir(parents=True)
    (p / "a.txt").write_text("0 0.9 0.5 0.5 0.2 0.2\n1 0.8 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def app(folder):
    """A live YoloLabeler on that folder; skipped when Tk has no display."""
    try:
        root = ctk.CTk()
    except Exception as e:
        pytest.skip(f"no Tk display: {e}")
    root.geometry("900x600")
    app = YoloLabeler(root)
    root.update()
    app._init_folder(str(folder))
    app._annotate_tab.load_image()
    root.update()
    yield app
    app._quit()


# ── annotate ────────────────────────────────────────────────────────────────

class TestAnnotate:
    def test_loads_document(self, app):
        assert app.document is not None
        assert [a.kind for a in app.document.annotations] == ["box"]
        assert app.mode == "box"

    def test_draw_delete_undo(self, app):
        tab = app._annotate_tab
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert len(app.document.annotations) == 2
        tab.undo_last()
        assert len(app.document.annotations) == 1
        tab.redo_last()
        assert len(app.document.annotations) == 2

    def test_save_and_reload(self, app):
        tab = app._annotate_tab
        tab.on_button_press(click_at(tab, 50, 50))
        tab.on_button_release(click_at(tab, 150, 150))
        assert tab.save_annotations() is None
        tab.load_image()
        assert len(app.document.annotations) == 2
        assert app.document.annotations[1].author == app._current_user

    def test_hidden_class_is_not_a_delete_target(self, app):
        app._select_class_by_id(0)
        app.class_names[1] = "other"
        app._select_class_by_id(1)
        assert app._annotate_tab.visible_annotations() == []
