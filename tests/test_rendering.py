"""Tests for yololabeler.rendering, halo text drawn on a real Tk canvas."""

import tkinter as tk

import pytest

from yololabeler.rendering import halo_text, _HALO_OFFSETS


@pytest.fixture(scope="module")
def tk_root():
    """One hidden Tk root for the module; skipped when there is no display."""
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def canvas(tk_root):
    """A real Tk Canvas, empty at the start of each test."""
    c = tk.Canvas(tk_root, width=200, height=200)
    yield c
    c.destroy()


# ── _HALO_OFFSETS ───────────────────────────────────────────────────────────

class TestHaloOffsets:
    def test_offsets_are_unique(self):
        assert len(set(_HALO_OFFSETS)) == len(_HALO_OFFSETS)

    def test_centre_is_not_an_offset(self):
        assert (0, 0) not in _HALO_OFFSETS

    def test_offsets_stay_within_two_pixels(self):
        assert all(abs(dx) <= 2 and abs(dy) <= 2 for dx, dy in _HALO_OFFSETS)


# ── halo_text ───────────────────────────────────────────────────────────────

class TestHaloText:
    def test_item_count(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        assert len(canvas.find_all()) == len(_HALO_OFFSETS) + 1

    def test_all_items_share_the_text(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        assert all(canvas.itemcget(i, "text") == "hello"
                   for i in canvas.find_all())

    def test_halo_items_are_black(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        items = canvas.find_all()
        assert all(canvas.itemcget(i, "fill") == "black" for i in items[:-1])

    def test_last_item_uses_the_fill_colour(self, canvas):
        halo_text(canvas, 50, 60, "hello", "#ff8800")
        last = canvas.find_all()[-1]
        assert canvas.itemcget(last, "fill") == "#ff8800"

    def test_last_item_sits_on_the_requested_point(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        last = canvas.find_all()[-1]
        assert canvas.coords(last) == [50.0, 60.0]

    def test_halo_positions_match_the_offsets(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white")
        items = canvas.find_all()
        positions = {tuple(canvas.coords(i)) for i in items[:-1]}
        expected = {(50.0 + dx, 60.0 + dy) for dx, dy in _HALO_OFFSETS}
        assert positions == expected

    def test_keyword_arguments_are_forwarded(self, canvas):
        halo_text(canvas, 50, 60, "hello", "white", anchor="nw")
        assert all(canvas.itemcget(i, "anchor") == "nw"
                   for i in canvas.find_all())

    def test_float_coordinates_accepted(self, canvas):
        halo_text(canvas, 10.5, 20.5, "x", "white")
        last = canvas.find_all()[-1]
        assert canvas.coords(last) == pytest.approx([10.5, 20.5])

    def test_repeated_calls_accumulate(self, canvas):
        halo_text(canvas, 10, 10, "a", "white")
        halo_text(canvas, 20, 20, "b", "white")
        assert len(canvas.find_all()) == 2 * (len(_HALO_OFFSETS) + 1)

    def test_empty_text_still_draws_items(self, canvas):
        halo_text(canvas, 10, 10, "", "white")
        assert len(canvas.find_all()) == len(_HALO_OFFSETS) + 1
